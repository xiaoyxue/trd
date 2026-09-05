//! The stdin Arrow-stream reader thread and the messages it forwards: the
//! decoded mesh table, optional bound texture, playback rate, and each frame.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use trd_core::{
    FrameParams, ImageData, ImageTexture, InlineFrameCache, InputStream, Mesh, MeshTableIndex,
    Scene, WireDraw,
};

/// A message from the stdin reader thread: the decoded CPU mesh table (sent
/// once, first), then the optional bound texture, then the playback rate,
/// then each decoded frame.
pub(crate) enum StreamMsg {
    Meshes {
        meshes: Vec<Mesh>,
        grid_mesh: Option<MeshTableIndex>,
    },
    // Only sent when the stream carries a texture table; small (width/height +
    // an RGBA byte buffer), so it needs no boxing.
    Texture(ImageTexture),
    Rate(f64),
    // Boxed: `FrameData` embeds the large `FrameParams` (camera columns), so an
    // unboxed variant would dwarf `Rate` (clippy::large_enum_variant).
    Frame(Box<FrameData>),
}

/// One decoded frame: its camera/transform params and validated wire draw
/// list, built into a [`trd_core::Scene`] at render time. `frame_image` holds a
/// per-frame background image (#63) decoded to RGBA at full source
/// resolution off the render thread (from `frame_path` + `--frames-base`),
/// uploaded + composited beneath the scene at render time (the GPU samples it
/// down to the surface via the frame plane's `Stretch` fit); `None` when the
/// frame has no background. It is an `Arc` so cloning a frame for playback never
/// re-copies the pixel buffer.
#[derive(Clone)]
pub(crate) struct FrameData {
    pub(crate) params: FrameParams,
    pub(crate) draws: Vec<WireDraw>,
    pub(crate) frame_image: Option<Arc<ImageData>>,
}

/// Reads the Arrow IPC frame-params stream from stdin on a background thread,
/// forwarding the stream's declared playback rate then each decoded frame over
/// `tx` until the stream ends. When `frames_base` is set, a frame's `frame_path`
/// is loaded and decoded to RGBA at full source resolution off the
/// render thread, then shipped with the frame for compositing (the GPU samples it
/// down to the surface via the frame plane's `Stretch` fit). Decoded stills are
/// held in an `Arc` so cloning a frame for loop playback never re-copies the
/// pixel buffer; per-frame decode is cheap because deps build with `opt-level=3`
/// even in dev (see the root `Cargo.toml`).
pub(crate) fn spawn_stdin_reader(
    tx: mpsc::Sender<StreamMsg>,
    frames_base: Option<PathBuf>,
    grid_mesh: Option<MeshTableIndex>,
) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = read_stream(std::io::stdin().lock(), &tx, frames_base, grid_mesh) {
                log::error!("input stream error: {err}");
            }
        });
    if let Err(err) = spawned {
        log::error!("failed to spawn stdin reader thread: {err}");
    }
}

/// Drives the stream: the prologue once, then a frame per timeline row.
///
/// The loop lives here rather than behind a callback API in `trd-core` because
/// it is three lines and this shell is the only thing that knows what to do with
/// each frame — forward it to the window thread, which paces playback itself.
fn read_stream<R: std::io::Read>(
    source: R,
    tx: &mpsc::Sender<StreamMsg>,
    frames_base: Option<PathBuf>,
    grid_mesh: Option<MeshTableIndex>,
) -> Result<(), trd_core::StreamError> {
    let mut input = InputStream::new(source);
    let prologue = input.prologue()?;
    let mesh_count = prologue.meshes.len();
    if let Some(row) = grid_mesh {
        Scene::validate_mesh_index(row, mesh_count)?;
    }
    let _ = tx.send(StreamMsg::Meshes {
        meshes: prologue.meshes.to_vec(),
        grid_mesh,
    });
    if let Some(texture) = prologue.texture {
        let _ = tx.send(StreamMsg::Texture(texture.clone()));
    }
    let _ = tx.send(StreamMsg::Rate(prologue.frame_rate));

    let mut inline_cache = InlineFrameCache::default();
    while let Some(batch) = input.next_batch() {
        for frame in batch? {
            let draws = frame.resolved_draws();
            Scene::validate_draws(&draws, mesh_count)?;
            let frame_image = inline_cache
                .resolve(frame.frame_id, input.frames())?
                .map(|(image, _changed)| image)
                .or_else(|| {
                    frame
                        .frame_ref
                        .as_deref()
                        .zip(frames_base.as_ref())
                        .and_then(|(rel, base)| load_frame_image(&base.join(rel)))
                        .map(Arc::new)
                });
            let _ = tx.send(StreamMsg::Frame(Box::new(FrameData {
                params: frame.params,
                draws,
                frame_image,
            })));
        }
    }
    input.finish()
}

/// Decodes a background frame image file (PNG/JPEG) to RGBA at its full source
/// resolution (#63). Kept in the shell so trd-core does no image I/O; a load
/// failure logs and yields `None` (that frame renders without a background).
fn load_frame_image(path: &std::path::Path) -> Option<ImageData> {
    match image::open(path) {
        Ok(img) => {
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();
            Some(ImageData {
                width,
                height,
                rgba: rgba.into_raw(),
            })
        }
        Err(err) => {
            log::warn!("skipping frame background {}: {err}", path.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trd_core::SceneError;

    fn fixture() -> Vec<u8> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("crates")
            .join("trd-core")
            .join("tests")
            .join("golden")
            .join("stage2.arrow");
        std::fs::read(path).unwrap()
    }

    #[test]
    fn reader_sends_cpu_meshes_once_and_validated_wire_draws() {
        let bytes = fixture();
        let (tx, rx) = mpsc::channel();
        read_stream(bytes.as_slice(), &tx, None, Some(MeshTableIndex::new(0))).unwrap();
        let StreamMsg::Meshes { meshes, grid_mesh } = rx.try_recv().unwrap() else {
            panic!("the CPU meshes must precede all frames");
        };
        assert_eq!(grid_mesh, Some(MeshTableIndex::new(0)));
        let received: Vec<_> = rx
            .try_iter()
            .filter_map(|message| match message {
                StreamMsg::Frame(frame) => Some(frame),
                StreamMsg::Meshes { .. } => panic!("meshes must be sent only once"),
                _ => None,
            })
            .collect();
        let mut input = trd_core::InputSession::new();
        let decoded: Vec<_> = input.push(&bytes).unwrap().into_iter().flatten().collect();
        input.finish().unwrap();
        assert_eq!(meshes.len(), input.meshes().len());
        assert!(!received.is_empty());
        assert_eq!(received.len(), decoded.len());
        for (received, wire) in received.iter().zip(&decoded) {
            assert_eq!(received.params, wire.params);
            assert_eq!(received.draws, wire.resolved_draws());
            Scene::validate_draws(&received.draws, meshes.len()).unwrap();
        }
    }

    #[test]
    fn reader_rejects_a_bad_grid_row_before_sending_resources_or_frames() {
        let bytes = fixture();
        let (tx, rx) = mpsc::channel();
        let error = read_stream(
            bytes.as_slice(),
            &tx,
            None,
            Some(MeshTableIndex::new(u32::MAX)),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            trd_core::StreamError::Scene(SceneError::MeshIndexOutOfRange { mesh_id, .. })
                if mesh_id == MeshTableIndex::new(u32::MAX)
        ));
        assert!(rx.try_iter().next().is_none());
    }
}
