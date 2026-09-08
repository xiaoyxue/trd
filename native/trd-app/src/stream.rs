//! Reads one retained `[params][mesh?]` document off the window thread.

use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};

use trd_core::{DocumentFrame, ImageData, MeshAsset, SceneDocument, StreamError, Tonemap};

pub(crate) enum StreamMsg {
    Meshes {
        assets: Vec<MeshAsset>,
        reference_only: bool,
    },
    Rate(f64),
    Tonemap(Tonemap),
    Frame(Box<FrameData>),
}

#[derive(Clone)]
pub(crate) struct FrameData {
    pub(crate) document: Arc<SceneDocument>,
    pub(crate) frame: DocumentFrame,
    pub(crate) frame_image: Option<Arc<ImageData>>,
}

pub(crate) fn spawn_stdin_reader(
    tx: mpsc::Sender<StreamMsg>,
    frames_base: Option<PathBuf>,
    wake: winit::event_loop::EventLoopProxy<()>,
) {
    let spawned = std::thread::Builder::new()
        .name("trd-stdin-reader".to_owned())
        .spawn(move || {
            let notify = || {
                let _ = wake.send_event(());
            };
            if let Err(error) = read_stream(
                std::io::stdin().lock(),
                &tx,
                frames_base.as_deref(),
                &notify,
            ) {
                log::error!("input stream error: {error}");
            }
            // Wake after disconnect too, so non-looping playback can become idle.
            drop(tx);
            notify();
        });
    if let Err(error) = spawned {
        log::error!("failed to spawn stdin reader thread: {error}");
    }
}

fn read_stream(
    input: impl Read,
    tx: &mpsc::Sender<StreamMsg>,
    frames_base: Option<&Path>,
    notify: &dyn Fn(),
) -> Result<(), StreamError> {
    let document = Arc::new(SceneDocument::read_from(input)?);
    let send = |message| {
        let sent = tx.send(message).is_ok();
        notify();
        sent
    };
    if !send(StreamMsg::Meshes {
        assets: document.decoded_assets()?,
        reference_only: document.meshes().is_empty(),
    }) || !send(StreamMsg::Rate(trd_core::frame_rate_from_metadata(
        document.schema().metadata(),
    ))) {
        return Ok(()); // The window closed.
    }
    if let Some(operator) = document.tonemap_override()? {
        if !send(StreamMsg::Tonemap(operator)) {
            return Ok(());
        }
    }
    let mut images = HashMap::<String, Arc<ImageData>>::new();
    for row in 0..document.row_count() {
        let frame_image = if let Some(reference) = document.frame_ref(row)? {
            if let Some(image) = images.get(&reference) {
                Some(Arc::clone(image))
            } else {
                let base =
                    frames_base.ok_or_else(|| StreamError::FrameResolve(reference.clone()))?;
                let image = Arc::new(load_frame_image(&base.join(&reference))?);
                images.insert(reference, Arc::clone(&image));
                Some(image)
            }
        } else {
            None
        };
        if !send(StreamMsg::Frame(Box::new(FrameData {
            frame: document.frame(row)?,
            document: Arc::clone(&document),
            frame_image,
        }))) {
            break;
        }
    }
    Ok(())
}

fn load_frame_image(path: &Path) -> Result<ImageData, StreamError> {
    let rgba = image::open(path)
        .map_err(|error| {
            std::io::Error::other(format!("read background {}: {error}", path.display()))
        })?
        .to_rgba8();
    Ok(ImageData {
        width: rgba.width(),
        height: rgba.height(),
        rgba: rgba.into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn current_document_reaches_the_window_without_changing_camera_or_glb_bindings() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/trd-core/tests/golden");
        let bytes = std::fs::read(root.join("stage2.arrow")).unwrap();
        let expected = SceneDocument::read(&bytes).unwrap();
        let (tx, rx) = mpsc::channel();
        let wakes = Cell::new(0);
        read_stream(&bytes[..], &tx, Some(&root), &|| wakes.set(wakes.get() + 1)).unwrap();
        drop(tx);
        let mut frames = Vec::new();
        let mut assets = None;
        for message in rx {
            match message {
                StreamMsg::Meshes {
                    assets: value,
                    reference_only,
                } => {
                    assert!(!reference_only);
                    assets = Some(value);
                }
                StreamMsg::Frame(frame) => frames.push(frame),
                StreamMsg::Rate(rate) => assert!(rate > 0.0),
                StreamMsg::Tonemap(_) => {}
            }
        }
        assert_eq!(assets.unwrap().len(), expected.meshes().len());
        assert_eq!(frames.len(), expected.row_count());
        assert!(wakes.get() >= frames.len() + 2);
        for (row, frame) in frames.iter().enumerate() {
            assert_eq!(frame.frame, expected.frame(row).unwrap());
            assert_eq!(frame.document.meshes(), expected.meshes());
            assert!(frame.frame_image.is_some());
        }
    }

    #[test]
    fn missing_background_is_an_error_not_an_incomplete_scene() {
        let bytes = include_bytes!("../../../crates/trd-core/tests/golden/stage1.arrow");
        let (tx, _rx) = mpsc::channel();
        assert!(matches!(
            read_stream(&bytes[..], &tx, None, &|| {}),
            Err(StreamError::FrameResolve(_))
        ));
    }
}
