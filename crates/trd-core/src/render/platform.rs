//! The genuinely platform-specific primitives, in one place (#180).
//!
//! Everything here exists because native and the browser differ in a way no
//! abstraction can paper over, and each item is **two or three lines** — far too
//! small to justify duplicating the algorithms that call them. Centralizing the
//! pairs keeps the `cfg` count down and, more usefully, makes the real surface
//! of platform difference in the render core inspectable at a glance: it is
//! exactly *instance creation* and *waiting for a buffer mapping*.
//!
//! **Not here on purpose:** task scheduling. `spawn_local` versus `block_on` is
//! an executor policy owned by whoever drives the UI loop, not by the renderer,
//! so it stays in the front-ends.

/// Creates the wgpu instance.
///
/// Native uses `new_without_display_handle_from_env()` so `WGPU_BACKEND` (e.g.
/// `gl` on WSL2, per `AGENTS.md`) is honoured; the browser has no such knob and
/// takes the default.
pub fn create_instance() -> wgpu::Instance {
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
    }
    #[cfg(target_arch = "wasm32")]
    {
        wgpu::Instance::default()
    }
}

/// Drives the device far enough for a pending `map_async` callback to run.
///
/// This is the one operation whose *shape* differs: natively the calling thread
/// may block until the mapping completes, so a `block_on`ed future is already
/// finished when it returns. In the browser blocking the event loop is not
/// allowed, so the queue is kicked once and the caller's `.await` yields control
/// back — the readback then completes on a later turn.
///
/// Both readback paths ([`TextureTarget`](super::TextureTarget) and
/// [`PickTarget`](super::PickTarget)) call this, so neither has to know which
/// platform it is on.
pub fn poll_for_map(device: &wgpu::Device) -> Result<(), wgpu::PollError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        device.poll(wgpu::PollType::wait_indefinitely()).map(|_| ())
    }
    #[cfg(target_arch = "wasm32")]
    {
        device.poll(wgpu::PollType::Poll).map(|_| ())
    }
}
