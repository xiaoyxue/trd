//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

mod math;
mod output;
mod protocol;
mod render;

pub use math::{
    Aabb2, Aabb3, Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Rotation, Scalar, ToWgsl,
    Transform, Vector2, Vector3, Vector4, EPSILON,
};
pub use output::{output_schema, tightly_pack_rgba, OutputError, OutputSession};
pub use protocol::{
    frame_rate_from_metadata, FrameBatch, InputSession, ProtocolError, DEFAULT_FRAME_RATE,
    FRAME_RATE_KEY, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY,
};
pub use render::{create_triangle_pipeline, render_triangle, FrameParams, TriangleRenderer};

#[cfg(not(target_arch = "wasm32"))]
mod stream;
#[cfg(not(target_arch = "wasm32"))]
pub use stream::{
    decode_frames, input_schema, read_frame_stream, read_frame_stream_with_meta, run_stream,
    StreamError,
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
