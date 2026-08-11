//! Shared, platform-agnostic mesh rendering.
//!
//! [`SceneRenderer`] rasterizes a [`Scene`] of [`DrawableObject`]s through the
//! vertex/index-buffer path used by the native batch renderer and the browser.

mod batch;
#[cfg(not(target_arch = "wasm32"))]
mod batch_renderer;
mod bound_material_maps;
mod bound_texture;
mod buffer;
mod color;
mod env_background;
mod frame_params;
mod frame_plane;
mod gizmo;
mod gpu_context;
mod gpu_types;
mod ibl;
mod light;
mod mesh_store;
mod options;
mod pbr;
mod picking;
mod pipeline;
mod render_target;
mod scene;
mod scene_renderer;
mod tonemap;
mod triangle_renderer;

#[cfg(test)]
mod gpu_tests;

// Public API surface (re-exported unchanged by `crate::lib`).
// The headless batch harness is native-only (drives wgpu under
// `pollster::block_on`), so it and its re-export are gated off wasm.
#[cfg(not(target_arch = "wasm32"))]
pub use batch_renderer::BatchRenderer;
pub use frame_params::{CameraFormError, FrameParams, Viewport};
pub use gpu_context::{
    create_instance, AdapterFacts, GpuContext, GpuInitError, GpuRequest, LimitsPreset,
};
pub use gpu_types::{Mesh, MeshShading, Vertex};
pub use ibl::{EnvMapData, ImageBasedLighting};
pub use light::{Light, Lighting, PointLight};
pub use options::{Msaa, PbrConfig, RenderOptions};
pub use pbr::PbrDebugView;
pub use picking::PickTarget;
pub use pipeline::create_mesh_pipeline;
pub use render_target::{
    OffscreenError, OffscreenTarget, OnscreenTarget, RenderTarget, OFFSCREEN_FORMAT,
};
pub use scene::{
    build_scene, plane_grid_overlays, selection_aabb_overlay, Draw, DrawableObject, FrameFit,
    GridPlane, RenderMode, Scene,
};
pub use scene_renderer::SceneRenderer;
pub use tonemap::{ToneMapping, Tonemap};
pub use triangle_renderer::TriangleRenderer;

// Crate-internal items shared across render submodules and sibling modules.
pub(crate) use color::upload_texture;
pub(crate) use frame_params::{projection_from_intrinsics, DEFAULT_FAR, DEFAULT_NEAR};
pub(crate) use gizmo::{
    aabb_line_vertices, axes_arrow_vertices, axes_line_vertices, blob_shadow_vertices,
    grid_line_vertices, quad_outline_vertices, SHADOW_VERTEX_COUNT,
};
pub(crate) use gpu_types::{
    GizmoLineVertex, GizmoUniform, InstanceRaw, PbrVertex, PickInstanceRaw, Uniform,
};
pub(crate) use ibl::BoundEnv;
pub(crate) use light::{DEFAULT_LIGHTS, DEFAULT_POINT_LIGHTS};
pub(crate) use pbr::{compute_smooth_normals, compute_tangents, PbrUniform, PbrUniformInputs};
pub(crate) use pipeline::{
    create_depth_target, create_env_bind_group_layout, create_frame_bind_group_layout,
    create_frame_plane_pipeline, create_gizmo_bind_group_layout, create_gizmo_binding,
    create_gizmo_line_pipeline, create_mesh_bind_group_layout, create_mesh_pipeline_with,
    create_msaa_color_target, create_pbr_bind_group_layout, create_pbr_pipeline,
    create_picking_pipeline, create_shadow_pipeline, create_texture_bind_group_layout,
    create_textured_pipeline, create_view_proj_binding, overlay_depth_stencil, solid_depth_stencil,
    write_gizmo_params, write_view_proj, DepthTarget, MsaaColorTarget, MSAA_SAMPLE_COUNT,
    PICK_FORMAT,
};
pub(crate) use scene::{frame_fit_uv_scale, DRAW_MODE_INHERIT};
