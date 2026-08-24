//! Error type for the trd-gui shell (issue #97).
//!
//! The GUI owns only UI, interaction, and scene authoring; every failure it can
//! raise is either a mesh-load problem (I/O + `trd-core` OBJ parse) or a render
//! problem delegated to `trd-core`. Rendering lives entirely in `trd-core`, so
//! the render arm only exists on native targets (where the in-process
//! [`crate::renderer`] is compiled).

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

    /// A texture image could not be decoded (PNG/JPEG). Shared: native
    /// `--texture` bytes and browser `?texture=` bytes both decode in Rust.
    #[error("failed to decode texture: {0}")]
    TextureDecode(#[from] image::error::ImageError),

    /// The decoded texture pixels were rejected by `trd-core`.
    #[error("invalid texture: {0}")]
    TextureData(#[from] trd_core::TextureError),

    /// A render delegated to `trd-core` failed (native in-process backend).
    #[cfg(not(target_arch = "wasm32"))]
    #[error("render failed: {0}")]
    Render(#[from] trd_core::StreamError),

    /// The shared `trd-core` renderer failed (construction or read-back). Its own
    /// error type since the renderer is platform-neutral (#180).
    #[error(transparent)]
    CoreRender(#[from] trd_core::RenderError),

    /// The texture image file could not be read from disk (native `--texture`).
    #[cfg(not(target_arch = "wasm32"))]
    #[error("failed to read texture file '{path}': {source}")]
    TextureIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// The HDR environment-map file could not be read from disk (native `--env`).
    #[cfg(not(target_arch = "wasm32"))]
    #[error("failed to read env-map file '{path}': {source}")]
    EnvIo {
        path: String,
        #[source]
        source: std::io::Error,
    },

    /// A picked model was rejected on size **before** being decoded (#353).
    #[error(
        "'{name}' is {:.1} MiB — over the {:.0} MiB limit for a loaded model",
        *size as f64 / (1024.0 * 1024.0),
        *limit as f64 / (1024.0 * 1024.0)
    )]
    ModelTooLarge {
        name: String,
        size: usize,
        limit: usize,
    },

    /// The picked file is not a glTF binary.
    #[error("'{name}' is not a GLB — a glTF binary starts with the magic \"glTF\"")]
    NotGlb { name: String },

    /// `trd-core` could not import the picked GLB.
    #[error("failed to import '{name}': {source}")]
    ModelImport {
        name: String,
        #[source]
        source: trd_core::GltfImportError,
    },

    /// A model is lit by the HDR probe alone, and the scene has none to bind.
    #[error("cannot load '{name}': no HDR environment probe is available to light it")]
    ModelNeedsEnvironment { name: String },
}
