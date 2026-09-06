//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

mod camera;
mod frame;
mod light;
mod material;
mod math;
mod media;
mod mesh;
// Byte transports live in `io/`; the sessions they drive stay with the format
// they implement, in `protocol/` (see `io/mod.rs`).
mod io;
mod protocol;
mod render;
mod session_state;
mod texture;

pub use camera::{Camera, DEFAULT_FIT_MARGIN, DEFAULT_FOV_Y, DEFAULT_VIEW_DIR};
pub use frame::{
    FrameError, InlineFrame, InlineFrameCache, FRAME_BYTES_COLUMN, FRAME_PIXELS_COLUMN,
};
// A directional light is universal domain vocabulary — zero wgpu — so it sits
// at the crate root beside `mesh`/`texture`/`camera`/`material` rather than in
// the render backend (#223).
pub use light::{EnvironmentLight, Light, Lighting, PointLight};
pub use math::{
    Aabb2, Aabb3, Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Rotation, Scalar, ToWgsl,
    Transform, Vector2, Vector3, Vector4, EPSILON,
};
// The CPU mesh, its three loaders (OBJ / Arrow / glTF) and the geometry derived
// from them are domain vocabulary, not renderer machinery, so they sit at the
// crate root beside `texture`/`camera`/`material` while their GPU residency
// stays in `render/` (#221/#224). Public paths are unchanged.
pub use mesh::{
    import_glb, import_gltf_materials, GltfAsset, GltfImportError, Mesh, MeshError, MeshId,
    MeshResourceError, MeshShading, MeshTableIndex, DEFAULT_PREVIEW_TARGET,
};

#[cfg(not(target_arch = "wasm32"))]
pub use io::{InputStream, Prologue};
pub use io::{OutputStream, SharedBuffer};
pub use protocol::{
    frame_rate_from_metadata, output_schema, read_image_stream, DecodedFrame, FrameBatch,
    InputSession, OutputError, OutputSession, ProtocolError, DEFAULT_FRAME_RATE, FRAME_RATE_KEY,
    PROTOCOL_VERSION, PROTOCOL_VERSION_KEY, TABLE_KIND_KEY,
};
// Material models are plain data (no wgpu, no bytemuck), so they sit beside
// `mesh`/`texture`/`camera` at the crate root rather than inside the render
// backend (#180). The public paths (`trd_core::DisneyMaterial`, ...) are unchanged.
pub use material::{AlphaMode, Auxiliary, DisneyMaterial, Material, MaterialTextures};
pub use render::{
    create_instance, AdapterFacts, CameraFormError, EnvMapData, ExternalFrame, FrameParams,
    GpuContext, GpuInitError, GpuRequest, ImageBasedLighting, MeshAppearance, MeshTarget, Msaa,
    PbrConfig, PbrDebugView, RenderOptions, RenderTarget, RenderTargetType, SceneLayer,
    SurfaceError, SurfaceRepair, SurfaceTarget, TargetError, TextureTarget, ToneMapping, Tonemap,
    Vertex, Viewport, TEXTURE_TARGET_FORMAT,
};
// The frame description (scene + primitives) is this rasterizer's own taxonomy
// — `Primitive` *is* the batch key (#204) — so it lives in `render/` beside the
// renderer that dispatches on it (#223). Public paths are unchanged.
pub use render::{
    Background, Draw, DrawSelection, DrawableObject, EnvironmentBackground, FrameFit, GridPlane,
    Primitive, RenderMode, Scene, SceneError, WireDraw,
};
// The render harness; available on both platforms since readback became async
// (#180) — the browser could not use it while it blocked on readback.
pub use media::{
    decode_video_editing_document, Shot, VideoEditingDocument, VideoEditingError,
    VideoEditingFrame, VIDEO_EDIT_TABLE_KIND_KEY, VIDEO_EDIT_TIMELINE_KIND, VIDEO_EDIT_VERSION,
    VIDEO_EDIT_VERSION_KEY,
};
pub use media::{probe_moov, UnpresentedTail, UnpresentedTailEvidence, VideoInfo, VideoTiming};
pub use render::{RenderError, Renderer};
pub use texture::{
    ConstantTexture, ImageData, ImageTexture, Texture, TextureError, TEXTURE_COLUMN,
};

#[cfg(not(target_arch = "wasm32"))]
mod stream_filter;
#[cfg(not(target_arch = "wasm32"))]
pub use stream_filter::{decode_frames, run_stream, FrameResolver, StreamError};

/// Returns the project greeting used by the CLI and web entry points.
pub fn greeting() -> String {
    "Hello from trd-core!".to_string()
}

/// CPU-only fixtures for downstream unit tests, not an upload/registration API.
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support {
    /// Fresh nonresident identities; a real renderer will reject them.
    pub fn mesh_ids(count: usize) -> Vec<crate::MeshId> {
        (0..count)
            .map(|_| crate::MeshId::fresh().expect("test identity space exhausted"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_mentions_trd() {
        assert!(greeting().contains("trd"));
    }
}
