//! Loading a model into a **live** scene (#353) — the one seam both delivery
//! surfaces call.
//!
//! The browser picks the file with an `<input type=file>` and the native window
//! with `rfd`, but neither parses anything: each hands over bytes plus the
//! file's name and this module does the rest — reject, decode, upload, register.
//!
//! Nothing here mutates the scene until the decode has succeeded, which is what
//! makes a bad file harmless: the renderer and [`SceneState`] are only touched
//! on the happy path, so the previous scene keeps rendering.

use crate::error::GuiError;
use crate::renderer::GuiRenderer;
use crate::scene::{ibl_only_lighting, SceneState};

/// The largest model accepted from a file picker.
///
/// Checked **before** decode, because the failure being guarded is a file big
/// enough to exhaust wasm's 32-bit address space while `gltf` builds its own
/// buffers — by the time that fails, the tab is already gone. The ceiling is
/// deliberately generous: real exports run well past 100 MiB, and the point is
/// to fail fast with a clear message instead of OOMing, not to police size.
///
/// 1 GiB is near the practical browser ceiling rather than a safe budget — the
/// decode holds the file *and* its expanded buffers and images at once, so a
/// model close to this may still exhaust a tab. It is the line past which we
/// refuse to try, not a promise that anything under it will load.
pub const MAX_MODEL_BYTES: usize = 1024 * 1024 * 1024;

/// The magic a glTF binary starts with.
const GLB_MAGIC: &[u8; 4] = b"glTF";

/// The output transform a glTF object is graded by — ACES at unit exposure,
/// matching what `?mesh=<glb>` already seeds so a model loaded at runtime and
/// one loaded at startup look the same.
fn gltf_tone_mapping() -> trd_core::ToneMapping {
    trd_core::ToneMapping {
        operator: trd_core::Tonemap::Aces,
        exposure: 1.0,
    }
}

/// A model a front-end has picked but not yet loaded: what the file picker
/// produced, in the form [`load_model`] consumes.
#[derive(Debug, Clone)]
pub struct PendingModel {
    /// The file's name, used only for messages.
    pub name: String,
    pub bytes: Vec<u8>,
    /// An HDR probe to bind if the scene has none, so the IBL-only rig the model
    /// is lit by has something to be lit by. Ignored when a probe is already
    /// bound.
    pub env_bytes: Option<Vec<u8>>,
}

/// Decodes `bytes` as a GLB, rejecting an oversized or non-GLB file first.
///
/// Split from [`load_model`] so the rejections are unit-testable without a GPU.
pub fn decode_glb(name: &str, bytes: &[u8]) -> Result<trd_core::GltfAsset, GuiError> {
    decode_glb_within(name, bytes, MAX_MODEL_BYTES)
}

/// [`decode_glb`] with an explicit ceiling, so the rejection can be exercised
/// without allocating a [`MAX_MODEL_BYTES`]-sized buffer to do it.
fn decode_glb_within(
    name: &str,
    bytes: &[u8],
    limit: usize,
) -> Result<trd_core::GltfAsset, GuiError> {
    if bytes.len() > limit {
        return Err(GuiError::ModelTooLarge {
            name: name.to_owned(),
            size: bytes.len(),
            limit,
        });
    }
    if !bytes.starts_with(GLB_MAGIC) {
        return Err(GuiError::NotGlb {
            name: name.to_owned(),
        });
    }
    trd_core::import_glb(bytes).map_err(|source| GuiError::ModelImport {
        name: name.to_owned(),
        source,
    })
}

/// Loads `request` into the live scene, returning the new object's index.
///
/// The model lands lit the way the video editor's Dragon is — [`RenderMode::Shaded`](trd_core::RenderMode::Shaded)
/// over the GLB's own material and maps, with an
/// [IBL-only rig](ibl_only_lighting) — so the same asset looks the same in both
/// front-ends.
///
/// On any error the scene is left exactly as it was.
pub fn load_model(
    renderer: &mut GuiRenderer,
    state: &mut SceneState,
    request: &PendingModel,
) -> Result<u32, GuiError> {
    let asset = decode_glb(&request.name, &request.bytes)?;

    // An IBL-only rig with no probe renders black, so the probe is bound first
    // and a failure to decode it aborts before anything is uploaded.
    if !renderer.has_env() {
        let Some(env_bytes) = request.env_bytes.as_ref() else {
            return Err(GuiError::ModelNeedsEnvironment {
                name: request.name.clone(),
            });
        };
        renderer.set_env(crate::assets::decode_env_hdr(env_bytes)?);
    }

    let mesh_id = renderer.add_model(&asset)?;
    let index = state.add_object(
        mesh_id,
        asset.material,
        trd_core::RenderMode::Shaded,
        gltf_tone_mapping(),
    );
    state.lighting = ibl_only_lighting();
    state.environment_available = true;
    Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_oversized_file_is_rejected_before_decode() {
        // A tiny explicit ceiling: the rejection is what is under test, not the
        // production constant, and allocating MAX_MODEL_BYTES to prove it would
        // cost a gigabyte.
        let error =
            decode_glb_within("huge.glb", &[0u8; 16], 8).expect_err("over the limit is rejected");
        assert!(
            matches!(error, GuiError::ModelTooLarge { size, limit, .. }
                if size == 16 && limit == 8),
            "expected a size rejection, got: {error}"
        );
        // Size is checked first: these bytes are not a GLB either, and the size
        // error is the one that must fire.
        assert!(
            error.to_string().contains("MiB"),
            "the message is in MiB, not raw bytes: {error}"
        );
    }

    /// The production ceiling is the one the panel advertises, and a file just
    /// under it is not rejected on size.
    #[test]
    fn the_limit_is_one_gibibyte() {
        assert_eq!(MAX_MODEL_BYTES, 1024 * 1024 * 1024);
        // Just under the limit falls through to the magic check, not the size one.
        let error = decode_glb_within("under.bin", b"not a glb", MAX_MODEL_BYTES)
            .expect_err("still not a GLB");
        assert!(matches!(error, GuiError::NotGlb { .. }));
    }

    #[test]
    fn a_non_glb_is_rejected_by_its_magic() {
        let error =
            decode_glb("notes.txt", b"v 0 0 0\nv 1 0 0\n").expect_err("an OBJ is not a GLB");
        assert!(
            matches!(error, GuiError::NotGlb { .. }),
            "expected a magic rejection, got: {error}"
        );
    }

    /// A file that *claims* to be a GLB but is truncated must surface the
    /// importer's own error rather than panicking.
    #[test]
    fn a_corrupt_glb_surfaces_the_import_error() {
        let error = decode_glb("broken.glb", b"glTF\x02\x00\x00\x00short")
            .expect_err("a truncated GLB fails to import");
        assert!(
            matches!(error, GuiError::ModelImport { .. }),
            "expected an import failure, got: {error}"
        );
    }

    /// The name is carried into every rejection, because "which file?" is the
    /// first thing the panel has to answer.
    #[test]
    fn rejections_name_the_file() {
        for (file, error) in [
            (
                "a.glb",
                decode_glb_within("a.glb", &[0u8; 4], 2).expect_err("too large"),
            ),
            (
                "b.obj",
                decode_glb("b.obj", b"not a glb").expect_err("not a glb"),
            ),
            (
                "c.glb",
                decode_glb("c.glb", b"glTF\x02\x00\x00\x00").expect_err("corrupt"),
            ),
        ] {
            let message = error.to_string();
            assert!(
                message.contains(file),
                "every rejection names its file, got: {message}"
            );
        }
    }
}
