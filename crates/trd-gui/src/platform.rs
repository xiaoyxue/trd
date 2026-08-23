//! The platform difference this front-end actually owns: task scheduling (#302).
//!
//! `trd-core`'s own shim says scheduling is "**not here on purpose** …  it stays
//! in the front-ends" (`render/platform.rs`). `trd-gui` *is* that front-end, so
//! the policy belongs here — once, rather than copied into every caller that
//! happens to await something.

/// Runs `future` to completion under this platform's executor.
///
/// **The two arms are not the same operation, and callers must not care.**
/// Natively this *blocks*; in the browser `spawn_local` returns immediately and
/// the future finishes on a later turn of the event loop. So the contract is
/// **"it will run; do not assume it has"** — every future's last act is to clear
/// its `*_in_flight` flag and request a repaint.
///
/// Named neither `spawn_local` nor `block_on`, because each would be a lie on
/// one of the two platforms.
pub(crate) fn drive(future: impl std::future::Future<Output = ()> + 'static) {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(future);
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(future);
}
