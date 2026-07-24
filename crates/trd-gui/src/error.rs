//! Error type for the trd-gui shell (issue #97).
//!
//! The GUI owns only UI, interaction, and scene authoring; every failure it can
//! raise is either a mesh-load problem (I/O + `trd-core` OBJ parse) or a render
//! problem delegated to `trd-core`. Rendering lives entirely in `trd-core`, so
//! the render arm only exists on native targets (where the in-process
//! [`crate::render_backend`] is compiled).

use thiserror::Error;

/// A failure in the trd-gui shell: loading the input mesh or rendering it.
#[derive(Debug, Error)]
pub enum GuiError {
    /// The mesh file could not be read from disk.
    #[error("failed to read mesh file '{path}': {source}")]
    MeshIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The mesh bytes could not be decoded as an OBJ by `trd-core`.
    #[error("failed to parse mesh: {0}")]
    Mesh(#[from] trd_core::MeshError),

    /// A render delegated to `trd-core` failed (native in-process backend).
    #[cfg(not(target_arch = "wasm32"))]
    #[error("render failed: {0}")]
    Render(#[from] trd_core::StreamError),

    /// The texture image file could not be read or decoded.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("failed to load texture '{path}': {source}")]
    TextureIo {
        path: String,
        #[source]
        source: image::error::ImageError,
    },

    /// The decoded texture pixels were rejected by `trd-core`.
    #[cfg(not(target_arch = "wasm32"))]
    #[error("invalid texture: {0}")]
    TextureData(#[from] trd_core::TextureError),
}
