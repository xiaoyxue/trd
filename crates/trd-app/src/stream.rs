//! The stdin Arrow-stream reader thread and the messages it forwards: the
//! decoded mesh table, optional bound texture, playback rate, and each frame.

use std::path::PathBuf;
use std::sync::mpsc;

use trd_core::{read_scene_stream_with_meta, Draw, FrameParams, ImageData, ImageTexture, Mesh};

/// A message from the stdin reader thread: the decoded mesh table (sent once,
/// first), then the optional bound texture (once, only for a `0.0.4` stream
/// carrying a texture table), then the stream's declared playback rate (once),
/// then each decoded frame.
pub(crate) enum StreamMsg {
    Meshes(Vec<Mesh>),
    // Only sent when the stream carries a texture table; small (width/height +
    // an RGBA byte buffer), so it needs no boxing.
    Texture(ImageTexture),
    Rate(f64),
    // Boxed: `FrameData` embeds the large `FrameParams` (camera columns), so an
    // unboxed variant would dwarf `Rate` (clippy::large_enum_variant).
    Frame(Box<FrameData>),
}

/// One decoded frame: its camera/transform params and resolved instanced draw
/// list, built into a [`trd_core::Scene`] at render time. `frame_image` holds a
/// per-frame background image (`0.0.5`, #63) already decoded to RGBA off the
/// render thread (from `frame_path` + `--frames-base`), uploaded + composited
/// beneath the scene at render time; `None` when the frame has no background.
#[derive(Clone)]
pub(crate) struct FrameData {
    pub(crate) params: FrameParams,
    pub(crate) draws: Vec<Draw>,
    pub(crate) frame_image: Option<ImageData>,
}

/// Reads the Arrow IPC frame-params stream from stdin on a background thread,
/// forwarding the stream's declared playback rate then each decoded frame over
/// `tx` until the stream ends. When `frames_base` is set, a frame's `frame_path`
/// (`0.0.5`, #63) is loaded + decoded to RGBA off the render thread and shipped
/// with the frame for compositing.
pub(crate) fn spawn_stdin_reader(tx: mpsc::Sender<StreamMsg>, frames_base: Option<PathBuf>) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            let stdin = std::io::stdin().lock();
            let meshes_tx = tx.clone();
            let texture_tx = tx.clone();
            let rate_tx = tx.clone();
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = read_scene_stream_with_meta(
                stdin,
                |meshes| {
                    let _ = meshes_tx.send(StreamMsg::Meshes(meshes));
                },
                |texture| {
                    if let Some(texture) = texture {
                        let _ = texture_tx.send(StreamMsg::Texture(texture));
                    }
                },
                |rate| {
                    let _ = rate_tx.send(StreamMsg::Rate(rate));
                },
                |params, draws, frame_ref| {
                    let frame_image = frame_ref
                        .as_deref()
                        .zip(frames_base.as_ref())
                        .and_then(|(rel, base)| load_frame_image(&base.join(rel)));
                    let _ = tx.send(StreamMsg::Frame(Box::new(FrameData {
                        params,
                        draws,
                        frame_image,
                    })));
                },
            ) {
                log::error!("input stream error: {err}");
            }
        });
    if let Err(err) = spawned {
        log::error!("failed to spawn stdin reader thread: {err}");
    }
}

/// Decodes a background frame image file (PNG/JPEG) to RGBA (#63). Kept in the
/// shell so trd-core does no image I/O; a load failure logs and yields `None`
/// (that frame renders without a background).
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
