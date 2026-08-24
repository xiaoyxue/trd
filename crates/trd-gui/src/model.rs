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
/// buffers — by the time that fails, the tab is already gone. 128 MiB clears the
/// demo assets (the Dragon GLB is ~7 MiB) with room to spare.
pub const MAX_MODEL_BYTES: usize = 128 * 1024 * 1024;

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
    if bytes.len() > MAX_MODEL_BYTES {
        return Err(GuiError::ModelTooLarge {
            name: name.to_owned(),
            size: bytes.len(),
            limit: MAX_MODEL_BYTES,
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

    let mesh_id = renderer.add_model(&asset);
    let index = state.add_object(
        asset.material,
        trd_core::RenderMode::Shaded,
        gltf_tone_mapping(),
    );
    debug_assert_eq!(
        mesh_id as u32, index,
        "the renderer's mesh id and the scene's object row must stay the same integer"
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
        // One byte over the limit, and not even a GLB: the size check must be
        // the one that fires, since it is the one that runs first.
        let error = decode_glb("huge.glb", &vec![0u8; MAX_MODEL_BYTES + 1])
            .expect_err("over the limit is rejected");
        assert!(
            matches!(error, GuiError::ModelTooLarge { size, limit, .. }
                if size == MAX_MODEL_BYTES + 1 && limit == MAX_MODEL_BYTES),
            "expected a size rejection, got: {error}"
        );
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
        for error in [
            decode_glb("a.glb", &vec![0u8; MAX_MODEL_BYTES + 1]).expect_err("too large"),
            decode_glb("b.obj", b"not a glb").expect_err("not a glb"),
            decode_glb("c.glb", b"glTF\x02\x00\x00\x00").expect_err("corrupt"),
        ] {
            let message = error.to_string();
            assert!(
                message.contains("a.glb") || message.contains("b.obj") || message.contains("c.glb"),
                "every rejection names its file, got: {message}"
            );
        }
    }
}
