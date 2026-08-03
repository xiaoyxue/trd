//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

mod camera;
mod math;
mod mesh;
mod output;
mod protocol;
mod render;
mod scene_encode;
mod texture;

pub use camera::{Camera, DEFAULT_FIT_MARGIN, DEFAULT_FOV_Y, DEFAULT_VIEW_DIR};
pub use math::{
    Aabb2, Aabb3, Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Rotation, Scalar, ToWgsl,
    Transform, Vector2, Vector3, Vector4, EPSILON,
};
pub use mesh::{MeshError, DEFAULT_PREVIEW_TARGET};
pub use output::{output_schema, read_image_stream, tightly_pack_rgba, OutputError, OutputSession};
pub use protocol::{
    decode_params_stream, frame_rate_from_metadata, DecodedFrame, FrameBatch, InputSession,
    ProtocolError, DEFAULT_FRAME_RATE, FRAME_RATE_KEY, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY,
};
pub use render::{
    build_scene, create_instance, create_mesh_pipeline, plane_grid_overlays, CameraFormError, Draw,
    DrawableObject, EnvMapData, FrameFit, FrameParams, GpuContext, GpuInitError, GpuRequest,
    GridPlane, LimitsPreset, Mesh, MeshRenderer, OffscreenError, OffscreenTarget, OnscreenTarget,
    PbrMaterial, RenderMode, Scene, Tonemap, TriangleRenderer, Vertex, Viewport, OFFSCREEN_FORMAT,
};
// The native-only headless batch harness, relocated from `stream` into `render`
// (#134); the crate-root re-export path (`trd_core::BatchRenderer`) is unchanged.
#[cfg(not(target_arch = "wasm32"))]
pub use render::BatchRenderer;
pub use scene_encode::{
    encode_mesh_stream, encode_params_stream, encode_scene, encode_texture_stream, SceneEncodeError,
};
pub use texture::{
    ConstantTexture, ImageData, ImageTexture, Texture, TextureError, TEXTURE_COLUMN,
};

#[cfg(not(target_arch = "wasm32"))]
mod stream;
#[cfg(not(target_arch = "wasm32"))]
pub use stream::{
    decode_frames, read_scene_stream_with_meta, run_stream, FrameResolver, Msaa, PbrConfig,
    RenderOptions, StreamError,
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
