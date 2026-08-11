//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

mod camera;
mod frame;
mod gltf;
mod material;
mod math;
mod mesh;
mod output;
mod protocol;
mod render;
mod scene_encode;
mod texture;
mod video_editing;

pub use camera::{Camera, DEFAULT_FIT_MARGIN, DEFAULT_FOV_Y, DEFAULT_VIEW_DIR};
pub use frame::{FrameError, InlineFrame, FRAME_BYTES_COLUMN, FRAME_PIXELS_COLUMN};
pub use gltf::{import_glb, import_gltf_materials, GltfAsset, GltfImportError};
pub use math::{
    Aabb2, Aabb3, Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Rotation, Scalar, ToWgsl,
    Transform, Vector2, Vector3, Vector4, EPSILON,
};
pub use mesh::{MeshError, DEFAULT_PREVIEW_TARGET};
pub use output::{output_schema, read_image_stream, tightly_pack_rgba, OutputError, OutputSession};
pub use protocol::{
    decode_params_stream, frame_rate_from_metadata, DecodedFrame, FrameBatch, InputSession,
    ProtocolError, DEFAULT_FRAME_RATE, FRAME_RATE_KEY, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY,
    TABLE_KIND_KEY,
};
// Material models are plain data (no wgpu, no bytemuck), so they sit beside
// `mesh`/`texture`/`camera` at the crate root rather than inside the render
// backend (#180). The public paths (`trd_core::DisneyMaterial`, ...) are unchanged.
pub use material::{AlphaMode, Auxiliary, DisneyMaterial, Material, MaterialTextures};
pub use render::{
    build_scene, create_instance, create_mesh_pipeline, plane_grid_overlays, scene_with_overlays,
    selection_aabb_overlay, AdapterFacts, CameraFormError, Draw, DrawableObject, EnvMapData,
    FrameFit, FrameParams, GpuContext, GpuInitError, GpuRequest, GridPlane, ImageBasedLighting,
    Light, Lighting, LimitsPreset, Mesh, MeshShading, Msaa, OffscreenError, OffscreenTarget,
    OnscreenTarget, PbrConfig, PbrDebugView, PickTarget, PointLight, PresentOutcome, RenderMode,
    RenderOptions, RenderTarget, Scene, SceneLayer, SceneRenderer, SurfaceSkip, ToneMapping,
    Tonemap, TriangleRenderer, Vertex, Viewport, OFFSCREEN_FORMAT,
};
// The offscreen harness; available on both platforms since it became async
// (#180) — the browser could not use it while it blocked on readback.
pub use render::{RenderError, Renderer};
pub use scene_encode::{
    encode_frames_stream, encode_mesh_stream, encode_params_stream,
    encode_params_stream_with_frame_ids, encode_scene, encode_scene_with_frames,
    encode_texture_stream, SceneEncodeError,
};
pub use texture::{
    ConstantTexture, ImageData, ImageTexture, Texture, TextureError, TEXTURE_COLUMN,
};
pub use video_editing::{
    decode_video_editing_document, VideoEditingDocument, VideoEditingError, VideoEditingFrame,
    VideoInfo, VIDEO_EDIT_TABLE_KIND_KEY, VIDEO_EDIT_TIMELINE_KIND, VIDEO_EDIT_VERSION,
    VIDEO_EDIT_VERSION_KEY,
};

#[cfg(not(target_arch = "wasm32"))]
mod stream;
#[cfg(not(target_arch = "wasm32"))]
pub use stream::{
    decode_frames, read_scene_stream_with_meta, run_stream, FrameResolver, StreamError,
};

/// Returns the project greeting used by the CLI and web entry points.
pub fn greeting() -> String {
    "Hello from trd-core!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_mentions_trd() {
        assert!(greeting().contains("trd"));
    }
}
