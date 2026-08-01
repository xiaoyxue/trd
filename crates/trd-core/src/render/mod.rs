//! Shared, platform-agnostic mesh rendering.
//!
//! [`MeshRenderer`] rasterizes a [`Scene`] of [`DrawableObject`]s through the
//! vertex/index-buffer path used by the native batch renderer and the browser.

#[cfg(not(target_arch = "wasm32"))]
mod batch_renderer;
mod bound_texture;
mod color;
mod frame_params;
mod frame_plane;
mod gizmo;
mod gpu_context;
mod gpu_types;
mod mesh_renderer;
mod offscreen;
mod onscreen;
mod pbr;
mod pipeline;
mod scene;
mod triangle_renderer;

#[cfg(test)]
mod gpu_tests;

// Public API surface (re-exported unchanged by `crate::lib`).
// The headless batch harness is native-only (drives wgpu under
// `pollster::block_on`), so it and its re-export are gated off wasm.
#[cfg(not(target_arch = "wasm32"))]
pub use batch_renderer::BatchRenderer;
pub use frame_params::{CameraFormError, FrameParams, Viewport};
pub use gpu_context::{create_instance, GpuContext, GpuInitError, GpuRequest, LimitsPreset};
pub use gpu_types::{Mesh, Vertex};
pub use mesh_renderer::MeshRenderer;
pub use offscreen::{OffscreenError, OffscreenTarget, OFFSCREEN_FORMAT};
pub use onscreen::OnscreenTarget;
pub use pbr::{EnvMapData, PbrMaterial, Tonemap};
pub use pipeline::create_mesh_pipeline;
pub use scene::{build_scene, Draw, DrawableObject, FrameFit, GridPlane, RenderMode, Scene};
pub use triangle_renderer::TriangleRenderer;

// Crate-internal items shared across render submodules and sibling modules.
pub(crate) use color::upload_texture;
pub(crate) use frame_params::{projection_from_intrinsics, DEFAULT_FAR, DEFAULT_NEAR};
pub(crate) use gizmo::{
    axes_vertices, blob_shadow_vertices, grid_vertices, AABB_COLOR, AABB_EDGE_INDICES,
    AXES_VERTEX_COUNT, GRID_VERTEX_COUNT, SHADOW_VERTEX_COUNT,
};
pub(crate) use gpu_types::{InstanceRaw, PbrVertex, Uniform};
pub(crate) use pbr::{compute_smooth_normals, BoundEnv, PbrUniform};
pub(crate) use pipeline::{
    create_depth_target, create_env_bind_group_layout, create_frame_bind_group_layout,
    create_frame_plane_pipeline, create_mesh_bind_group_layout, create_mesh_pipeline_with,
    create_msaa_color_target, create_pbr_bind_group_layout, create_pbr_pipeline,
    create_shadow_pipeline, create_texture_bind_group_layout, create_textured_pipeline,
    create_view_proj_binding, overlay_depth_stencil, solid_depth_stencil, write_view_proj,
    DepthTarget, MsaaColorTarget, MSAA_SAMPLE_COUNT,
};
pub(crate) use scene::{frame_fit_uv_scale, DRAW_MODE_INHERIT};
