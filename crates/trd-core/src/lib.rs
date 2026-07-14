//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

mod protocol;
mod render;

pub use protocol::{
    FrameBatch, InputSession, ProtocolError, PROTOCOL_VERSION, PROTOCOL_VERSION_KEY,
};
pub use render::{create_triangle_pipeline, render_triangle, FrameParams, TriangleRenderer};

#[cfg(not(target_arch = "wasm32"))]
mod stream;
#[cfg(not(target_arch = "wasm32"))]
pub use stream::{
    decode_frames, input_schema, output_schema, read_frame_stream, run_stream, StreamError,
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
