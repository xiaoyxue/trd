//! The platform difference this front-end actually owns: task scheduling (#302).
//!
//! `trd-core`'s own shim says scheduling is "**not here on purpose** …  it stays
//! in the front-ends" (`render/platform.rs`). `trd-gui` *is* that front-end, so
//! the policy belongs here — once, rather than copied into every caller that
//! happens to await something.

/// Runs `future` to completion under this platform's executor.
///
/// **The two arms are not the same operation, and callers must not care.**
/// Natively this *blocks*, so anything after the call observes the future's
/// effects. In the browser it *detaches*: `spawn_local` returns immediately and
/// the future finishes on a later turn of the event loop. Every caller already
/// tolerates both — each future's last act is to clear its `*_in_flight` flag
/// and request a repaint, which is exactly how the browser's completion becomes
/// visible — so the contract is "it will run; do not assume it has".
///
/// The name is deliberately neither of the two: `spawn_local` would be a lie
/// natively and `block_on` a lie in the browser, and that asymmetry went
/// unstated at both of the call sites this replaces.
pub(crate) fn drive(future: impl std::future::Future<Output = ()> + 'static) {
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_futures::spawn_local(future);
    #[cfg(not(target_arch = "wasm32"))]
    pollster::block_on(future);
}
