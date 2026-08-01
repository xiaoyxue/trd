//! Errors raised while setting up the window or GPU.

/// Errors that can occur while setting up the window or GPU.
#[derive(Debug, thiserror::Error)]
pub enum AppError {
    /// The winit event loop could not be created.
    #[error("failed to create the event loop: {0}")]
    EventLoop(#[from] winit::error::EventLoopError),
    /// A wgpu surface could not be created from the window.
    #[error("failed to create a GPU surface: {0}")]
    CreateSurface(#[from] wgpu::CreateSurfaceError),
    /// The shared `trd-core` device/adapter helper failed (no adapter or the
    /// device could not be created).
    #[error(transparent)]
    Gpu(#[from] trd_core::GpuInitError),
    /// The adapter does not support the window surface.
    #[error("the GPU adapter does not support the window surface")]
    SurfaceUnsupported,
    /// The `--env` HDR environment map could not be decoded.
    #[error("failed to load environment map: {0}")]
    EnvMap(String),
}
