//! Shared, platform-agnostic mesh rendering.
//!
//! [`Renderer`] rasterizes a [`Scene`] of [`DrawableObject`]s through the
//! vertex/index-buffer path used by every native and browser front-end (#203).

mod bound_material_maps;
mod bound_texture;
mod bound_uniform;
mod buffer;
mod color;
mod draw_command;
mod env_map;
mod environment;
mod frame_params;
mod frame_plane;
mod gizmo;
mod gpu_context;
mod gpu_types;
mod light;
mod mesh_store;
mod options;
mod pbr;
mod picking;
mod pipeline;
mod platform;
mod render_target;
mod renderer;
mod scene_pipelines;
mod tonemap;
#[cfg(test)]
mod triangle_renderer;

#[cfg(test)]
mod gpu_tests;

// Public API surface (re-exported unchanged by `crate::lib`).
// The headless offscreen harness is native-only (drives wgpu under
// `pollster::block_on`), so it and its re-export are gated off wasm.
pub use env_map::{EnvMapData, ImageBasedLighting};
pub use frame_params::{CameraFormError, FrameParams, Viewport};
pub(crate) use gpu_context::LimitsPreset;
pub use gpu_context::{create_instance, AdapterFacts, GpuContext, GpuInitError, GpuRequest};
pub use gpu_types::Vertex;
pub use light::{Light, Lighting, PointLight};
pub use options::{Msaa, PbrConfig, RenderOptions};
pub use pbr::PbrDebugView;
pub use picking::PickTarget;
pub use render_target::{
    RenderTarget, RenderTargetType, SceneLayer, SurfaceTarget, TargetError, TextureTarget,
    TEXTURE_TARGET_FORMAT,
};
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use renderer::check_dimensions;
pub use renderer::{RenderError, Renderer, SurfaceError, SurfaceRepair};
pub use tonemap::{ToneMapping, Tonemap};
/// Reference + test scaffolding only (#202): the minimal canonical wgpu
/// renderer, kept to be *read* and exercised by `render::gpu_tests`. It has no
/// production consumer, so it is compiled for tests only.
#[cfg(test)]
pub(crate) use triangle_renderer::TriangleRenderer;

// Crate-internal items shared across render submodules and sibling modules.
pub(crate) use bound_uniform::{BoundUniform, BoundUniformArray};
pub(crate) use color::upload_texture;
// The CPU mesh is domain vocabulary and lives at the crate root (#221); it is
// re-exported here so `render/`'s `use super::*` glob imports keep resolving.
pub(crate) use crate::mesh::Mesh;
pub(crate) use frame_params::{projection_from_intrinsics, DEFAULT_FAR, DEFAULT_NEAR};
pub(crate) use gizmo::{
    aabb_line_vertices, axes_arrow_vertices, axes_line_vertices, blob_shadow_vertices,
    grid_line_vertices, quad_outline_vertices, SHADOW_VERTEX_COUNT,
};
pub(crate) use gpu_types::{
    GizmoLineVertex, GizmoUniform, InstanceRaw, PbrVertex, PickInstanceRaw, Uniform,
};
pub(crate) use light::{DEFAULT_LIGHTS, DEFAULT_POINT_LIGHTS};
pub(crate) use pbr::{compute_smooth_normals, compute_tangents, PbrUniform, PbrUniformInputs};
pub(crate) use pipeline::{
    create_depth_target, create_env_bind_group_layout, create_frame_bind_group_layout,
    create_frame_plane_pipeline, create_gizmo_bind_group_layout, create_gizmo_binding,
    create_gizmo_line_pipeline, create_mesh_bind_group_layout, create_mesh_pipeline_with,
    create_pbr_bind_group_layout, create_pbr_pipeline, create_picking_pipeline,
    create_shadow_pipeline, create_texture_bind_group_layout, create_textured_pipeline,
    create_view_proj_binding, overlay_depth_stencil, solid_depth_stencil, write_gizmo_params,
    write_view_proj, DepthTarget, MsaaColor, MSAA_SAMPLE_COUNT, PICK_FORMAT,
};
pub(crate) use scene_pipelines::{create_scene_pipelines, ScenePipelines, SceneUniforms};
