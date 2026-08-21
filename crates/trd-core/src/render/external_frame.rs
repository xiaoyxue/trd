//! The seam for a frame whose pixels are **already on the GPU** (#229, #302).
//!
//! The background frame plane owns the destination texture but has no business
//! knowing where pixels come from. Usually it does not need to: `upload_rgba`
//! takes bytes, and bytes are bytes on every platform. A browser frame is the
//! exception, and not by preference — `Queue::copy_external_image_to_texture` is
//! `#[cfg(web)]` in wgpu (`wgpu-30.0.0/src/api/queue.rs:264`), so the call that
//! performs the copy **cannot be compiled** into a crate that also builds
//! natively.
//!
//! That constrains *where the implementation may live*, which is the whole
//! point. It does not constrain the seam: `trd-core` names this trait, allocates
//! the texture, and keeps its own invariants (size, format, the
//! `RENDER_ATTACHMENT` usage the external copy requires); the delivery surface
//! that decoded the frame supplies **only** the copy. `crates/trd-wasm` is the
//! one implementor today, over a WebCodecs `VideoFrame`.
//!
//! The alternative — `#[cfg(target_arch = "wasm32")]` around a `web_sys` type in
//! the platform-neutral render core — is what this replaces: eleven `cfg`s and a
//! `web-sys` dependency across two shared crates, none of it visible to a native
//! build and so none of it checked by one.

/// A decoded frame the delivery surface kept in GPU memory.
///
/// Implement this to hand `trd-core` a frame it must not download: the browser
/// has already decoded it, in hardware where it can, into GPU memory, and the
/// RGBA route drags it back down three times — `VideoFrame.copyTo` (which also
/// does the YUV→RGBA conversion), the wasm-bindgen boundary, and
/// `write_texture` — at *source* resolution, ~99 MB per frame for 4K (#229).
///
/// **Zero-copy is the browser's decision, not this trait's.** The spec does not
/// guarantee it; a software-decoded frame starts in CPU memory and a YUV→RGB
/// pass may still run. What implementing this guarantees is that *trd* no longer
/// forces the download.
pub trait ExternalFrame {
    /// The frame's dimensions in pixels, both non-zero.
    ///
    /// The frame is authoritative for its own size — the caller does not pass
    /// one alongside — so this is what the destination texture is allocated to
    /// and what the copy extent is taken from. A zero here is a caller bug and
    /// panics rather than producing a silently blank frame.
    fn size(&self) -> (u32, u32);

    /// Copies this frame's pixels into `texture`, which the caller has already
    /// allocated at [`size`](Self::size) with `RENDER_ATTACHMENT` usage.
    ///
    /// Implementations schedule the copy on `queue` and return; they do not own
    /// the texture and must not reallocate or reconfigure it. WebGPU snapshots
    /// the source during the call, so an implementor is free to release its
    /// frame afterwards — though nothing here requires it, and trd's own
    /// browser frame is instead held until a newer one supersedes it, because a
    /// repaint re-renders the same frame.
    fn copy_into(&self, queue: &wgpu::Queue, texture: &wgpu::Texture);
}
