//! Shared, platform-agnostic mesh rendering — **this rasterizer**: its frame
//! description, its primitive taxonomy, its pipelines and its GPU resources.
//!
//! [`Renderer`] rasterizes a [`Scene`] of [`DrawableObject`]s through the
//! vertex/index-buffer path used by every native and browser front-end (#203).
//!
//! The **frame description** lives here too (#223). It used to be `src/visual/`,
//! a module whose members shared no predicate beyond "not the renderer" — but
//! `Primitive::{Mesh, AabbBox, PlaneGrid, QuadOutline, BlobShadow,
//! CoordinateAxes}` and `RenderMode::{Filled, Textured, Shaded, Wireframe}` are
//! not general graphics vocabulary, they are the taxonomy [`Renderer`] dispatches
//! on, deliberately in lockstep with the batcher (#204). A different renderer
//! would not reuse them, so they sit beside the renderer they describe:
//!
//! | module | owns |
//! |---|---|
//! | [`scene`] | [`Scene`], its [`Background`], and its assembly, [`Scene::from_draws`] |
//! | [`drawable`] | [`Primitive`] — *what* can be drawn — and [`DrawableObject`], one placed by a model |
//! | [`draw`] | [`Draw`] + [`DrawSelection`], the *wire* instance record and its byte codec |
//! | [`draw_config`] | [`RenderMode`], [`FrameFit`], [`GridPlane`] — the per-drawable configuration a front-end selects |
//!
//! Assembly ([`Scene::from_draws`]) is the **one** place a wire [`Draw`] becomes
//! a [`DrawableObject`], which is what keeps every front-end rendering the same
//! scene from the same inputs (#180).
//!
//! Nothing in those four files touches wgpu, and the guarantee is carried by the
//! **derives**: a `wgpu::BindGroup`/`RenderPipeline`/`Buffer` field on [`Scene`]
//! would break `#[derive(Clone, Default, PartialEq)]` at compile time, since none
//! of them implement `PartialEq`. Treat any PR dropping those derives as removing
//! the guard.

mod bound_material_maps;
mod bound_texture;
mod bound_uniform;
mod buffer;
mod color;
mod draw;
mod draw_command;
mod draw_config;
mod drawable;
mod env_map;
mod environment;
mod frame_params;
mod frame_plane;
mod gizmo;
mod gpu_context;
mod gpu_types;
mod mesh_store;
mod options;
mod pbr;
mod picking;
mod pipeline;
mod platform;
mod render_pipelines;
mod render_target;
mod renderer;
mod scene;
mod tonemap;
#[cfg(test)]
mod triangle_renderer;

#[cfg(test)]
mod gpu_tests;

// Public API surface (re-exported unchanged by `crate::lib`).
// The headless offscreen harness is native-only (drives wgpu under
// `pollster::block_on`), so it and its re-export are gated off wasm.
pub use draw::{Draw, DrawSelection};
pub use draw_config::{FrameFit, GridPlane, RenderMode};
pub use drawable::{DrawableObject, Primitive};
pub use env_map::{EnvMapData, ImageBasedLighting};
pub use frame_params::{CameraFormError, FrameParams, Viewport};
pub(crate) use gpu_context::LimitsPreset;
pub use gpu_context::{create_instance, AdapterFacts, GpuContext, GpuInitError, GpuRequest};
pub use gpu_types::Vertex;
pub use options::{Msaa, PbrConfig, RenderOptions};
pub use pbr::PbrDebugView;
pub use render_target::{
    RenderTarget, RenderTargetType, SceneLayer, SurfaceTarget, TargetError, TextureTarget,
    TEXTURE_TARGET_FORMAT,
};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use renderer::check_dimensions;
pub use renderer::{RenderError, Renderer, SurfaceError, SurfaceRepair};
pub use scene::{Background, EnvironmentBackground, Scene};
pub use tonemap::{ToneMapping, Tonemap};
/// Reference + test scaffolding only (#202): the minimal canonical wgpu
/// renderer, kept to be *read* and exercised by `render::gpu_tests`. It has no
/// production consumer, so it is compiled for tests only.
#[cfg(test)]
pub(crate) use triangle_renderer::TriangleRenderer;

// Crate-internal items shared across render submodules and sibling modules.
pub(crate) use bound_uniform::{BoundSceneSlots, BoundUniform};
pub(crate) use color::upload_texture;
// The CPU mesh is domain vocabulary and lives at the crate root (#221); it is
// re-exported here so `render/`'s `use super::*` glob imports keep resolving.
pub(crate) use crate::mesh::Mesh;
pub(crate) use frame_params::{projection_from_intrinsics, DEFAULT_FAR, DEFAULT_NEAR};
// Only the AABB generator (used per uploaded mesh) and the shadow vertex count
// leave `gizmo.rs` now: the constant gizmo buffers are built inside
// `GizmoGeometry`, beside the generators that fill them (#222).
pub(crate) use gizmo::aabb_line_vertices;
pub(crate) use gpu_types::{
    GizmoLineVertex, GizmoUniform, InstanceRaw, PbrVertex, PickInstanceRaw, Uniform,
};
// The light rig is universal domain vocabulary and lives at the crate root
// (#223); re-exported here so `render/`'s `use super::*` globs keep resolving.
pub(crate) use crate::light::{Lighting, DEFAULT_LIGHTS, DEFAULT_POINT_LIGHTS};
#[cfg(test)]
pub(crate) use draw::DRAW_MODE_INHERIT;
pub(crate) use draw_config::frame_fit_uv_scale;
pub(crate) use pbr::{
    compute_smooth_normals, compute_tangents, PbrSceneUniform, PbrUniform, PbrUniformInputs,
};
pub(crate) use pipeline::{
    create_depth_target, create_env_bind_group_layout, create_frame_bind_group_layout,
    create_frame_plane_pipeline, create_gizmo_bind_group_layout, create_gizmo_binding,
    create_gizmo_line_pipeline, create_mesh_bind_group_layout, create_mesh_pipeline_with,
    create_pbr_bind_group_layout, create_pbr_pipeline, create_picking_pipeline,
    create_shadow_pipeline, create_texture_bind_group_layout, create_textured_pipeline,
    create_view_proj_binding, overlay_depth_stencil, solid_depth_stencil, write_gizmo_params,
    write_view_proj, DepthTarget, MsaaColor, MSAA_SAMPLE_COUNT, PICK_FORMAT,
};
pub(crate) use render_pipelines::{create_render_pipelines, RenderPipelines, SceneUniforms};
