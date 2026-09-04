//! The stdin Arrow-stream reader thread and the messages it forwards: the
//! decoded mesh table, optional bound texture, playback rate, and each frame.

use std::io::Read;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;

use trd_core::{
    Draw, FrameParams, ImageData, InlineFrameCache, InputStream, Mesh, MeshAsset, SceneError,
    Tonemap,
};

/// A message from the stdin reader thread: the decoded mesh table (sent once,
/// first), then the optional bound texture (once, only for a `0.0.4` stream
/// carrying a texture table), then the stream's declared playback rate (once),
/// then each decoded frame.
pub(crate) enum StreamMsg {
    Meshes(Vec<Mesh>),
    MeshAssets(Vec<MeshAsset>),
    Rate(f64),
    Tonemap(Tonemap),
    // Boxed: `FrameData` embeds the large `FrameParams` (camera columns), so an
    // unboxed variant would dwarf `Rate` (clippy::large_enum_variant).
    Frame(Box<FrameData>),
}

/// One decoded frame: its camera/transform params and resolved instanced draw
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
    pub(crate) draws: Vec<Draw>,
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
pub(crate) fn spawn_stdin_reader(tx: mpsc::Sender<StreamMsg>, frames_base: Option<PathBuf>) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_string())
        .spawn(move || {
            // A send error just means the window closed; stop reading in that case.
            if let Err(err) = read_stdin(&tx, frames_base) {
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
fn read_stdin(
    tx: &mpsc::Sender<StreamMsg>,
    frames_base: Option<PathBuf>,
) -> Result<(), trd_core::StreamError> {
    let mut input = InputStream::new(std::io::stdin().lock());
    let references = input.prologue()?.mesh_references;
    for (index, reference) in references {
        let bytes = load_mesh_reference(&reference).map_err(|message| {
            trd_core::StreamError::MeshResolve {
                index,
                reference: reference.display().to_owned(),
                message,
            }
        })?;
        input.resolve_gltf(index, &bytes)?;
    }
    let prologue = input.prologue()?;
    let mesh_count = prologue.meshes.len();
    let _ = tx.send(StreamMsg::Meshes(prologue.meshes.to_vec()));
    let _ = tx.send(StreamMsg::MeshAssets(prologue.mesh_assets.to_vec()));
    let _ = tx.send(StreamMsg::Rate(prologue.frame_rate));
    if let Some(operator) = input.tonemap_override() {
        let _ = tx.send(StreamMsg::Tonemap(operator));
    }

    let mut inline_cache = InlineFrameCache::default();
    while let Some(batch) = input.next_batch() {
        for frame in batch? {
            let draws = frame.resolved_draws();
            if let Some(bad) = draws.iter().find(|d| d.mesh_id as usize >= mesh_count) {
                return Err(SceneError::MeshIndexOutOfRange {
                    mesh_id: bad.mesh_id,
                    mesh_count,
                }
                .into());
            }
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

fn load_mesh_reference(reference: &trd_core::MeshReference) -> Result<Vec<u8>, String> {
    if let Some(path) = reference.path.as_ref() {
        match std::fs::read(path) {
            Ok(bytes) => return Ok(bytes),
            Err(error) if reference.url.is_none() => {
                return Err(format!("failed to read {path}: {error}"));
            }
            Err(_) => {}
        }
    }
    let url = reference
        .url
        .as_deref()
        .ok_or_else(|| "glTF reference has neither a readable path nor a URL".to_owned())?;
    let mut response = ureq::get(url)
        .call()
        .map_err(|error| format!("{url}: {error}"))?;
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(256 * 1024 * 1024)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("{url}: {error}"))?;
    Ok(bytes)
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
