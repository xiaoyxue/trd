//! Byte transports: the types that own a [`Read`](std::io::Read) or a
//! [`Write`](std::io::Write).
//!
//! The split against `protocol/` is exactly the one the type names draw: a type
//! that **owns** a transport is a `*Stream` and lives here; one that owns none is
//! a `*Session` and stays with the format it implements. Both directions nest
//! the same way, with the generic parameter on the outer layer so the semantic
//! layer is never polluted by a transport it should not know about:
//!
//! ```text
//! InputStream<R: Read>   ⊃  InputSession     (protocol/, cross-platform)
//! OutputStream<W: Write> ⊃  OutputSession    (protocol/, cross-platform)
//! ```
//!
//! Only [`InputStream`] is native-only. A browser has no `R: Read` — `read`
//! returning `Ok(0)` means EOF by contract, and "not arrived yet" is not EOF —
//! so wasm drives `InputSession` directly from its event loop. The output side
//! has just one execution model (push), and `Vec<u8>` already implements
//! `Write`, so `OutputStream<W>` serves both platforms.

#[cfg(not(target_arch = "wasm32"))]
mod input_stream;
mod output_stream;

#[cfg(not(target_arch = "wasm32"))]
pub use input_stream::{InputStream, Prologue};
pub use output_stream::{OutputStream, SharedBuffer};
