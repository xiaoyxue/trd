//! trd-core: the platform-agnostic wgpu rendering core.
//!
//! The same rendering logic runs natively (CLI) and in the browser (wasm).
//! Native and web entry points are thin wrappers that only provide a render
//! target and call into this crate.

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
