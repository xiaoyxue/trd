//! The `trd-gui` JS ABI: the interactive viewer (`start`) and the video editor
//! (`startVideoEditing` + [`VideoEditingHandle`]).
//!
//! `trd-gui` itself is a plain rlib — all UI, interaction, scene authoring, and
//! rendering live there, free of `wasm-bindgen`. **Every** `#[wasm_bindgen]`
//! export in the repo lives in this crate (#180), so there is exactly one browser
//! delivery surface to build and one generated JS package to import.

use std::rc::Rc;

/// The browser entry point (Slice 4): builds the offscreen renderer and runs the
/// eframe app on `canvas`. `mesh_bytes`, `texture_bytes`, and `env_bytes` are the
/// browser equivalents of the native `--mesh` / `--texture` / `--env` flags — an
/// optional Wavefront OBJ or binary glTF **as bytes**, optional texture image
/// **bytes** (PNG/JPEG), and an optional Radiance HDR environment probe **bytes**; the thin
/// JS bootstrap fetches them from `?mesh=` / `?texture=` / `?env=` URLs and passes
/// them in. `None`/absent falls back to the built-in cube / no texture / no probe.
/// Supplying an env probe starts the viewer in Disney **PBR** mode (the material
/// is then editable live in the UI). All UI + interaction + rendering happen in
/// Rust, per the repo's "JS is a thin bootstrap only" invariant.
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    mesh_bytes: js_sys::Array,
    texture_bytes: js_sys::Array,
    env_bytes: Option<Vec<u8>>,
) -> Result<(), wasm_bindgen::JsValue> {
    use crate::gui_web_app::WebApp;
    use trd_gui::interaction::InteractionController;
    use trd_gui::renderer::{GuiRenderer, MaterialMaps};
    use trd_gui::scene::{SceneSeed, SceneState};

    console_error_panic_hook::set_once();
    let _ = eframe::WebLogger::init(log::LevelFilter::Warn);

    let to_js = |e: trd_gui::error::GuiError| wasm_bindgen::JsValue::from_str(&e.to_string());
    struct LoadedMesh {
        mesh: trd_core::Mesh,
        material: trd_core::DisneyMaterial,
        texture: Option<trd_core::ImageTexture>,
        metallic_roughness: Option<trd_core::ImageTexture>,
        normal: Option<trd_core::ImageTexture>,
        is_gltf: bool,
    }

    /// Starts the dedicated `web/gui-video-editing/` poster/document example.
    /// Starts the dedicated `web/gui-video-editing/` editor.
    ///
    /// `document_bytes` is **optional**: without one the editor is a plain
    /// player — the timeline comes from the container (see
    /// `setVideoTimelineFromMoov`) and the placement UI stays inert (#264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = startVideoEditing)]
    pub async fn start_video_editing(
        canvas: web_sys::HtmlCanvasElement,
        document_bytes: Option<Vec<u8>>,
    ) -> Result<VideoEditingHandle, wasm_bindgen::JsValue> {
        use std::rc::Rc;

        console_error_panic_hook::set_once();
        let _ = eframe::WebLogger::init(log::LevelFilter::Warn);
        let document = document_bytes
            .map(|bytes| trd_core::decode_video_editing_document(&bytes))
            .transpose()
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
        let shared = Rc::new(trd_gui::video_editing::VideoEditingShared::default());
        let handle = match document.as_ref() {
            Some(document) => VideoEditingHandle::new(document, shared.clone()),
            None => VideoEditingHandle::player(shared.clone()),
        };
        // A placeholder until a video is opened: the real size arrives with the
        // container probe, and the target is resized to the fitted panel anyway.
        let (width, height) = document.as_ref().map_or((1280, 720), |document| {
            (document.video.width, document.video.height)
        });
        let creator_shared = shared.clone();
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                // Built **inside** the creator so the renderer can adopt eframe's
                // own WebGPU device: one device means trd's rendered texture is
                // bound straight into egui, with no GPU→CPU→GPU round trip per
                // frame. `wgpu_render_state` exists nowhere else — which is why
                // the renderer used to be built before `start`, and therefore had
                // to request a device of its own.
                Box::new(move |context| {
                    let state = context
                        .wgpu_render_state
                        .as_ref()
                        .ok_or("eframe has no wgpu render state")?;
                    let gpu = trd_core::GpuContext::adopt(
                        state.adapter.clone(),
                        state.device.clone(),
                        state.queue.clone(),
                    );
                    let renderer =
                        trd_gui::video_editing_renderer::VideoPlacementRenderer::new_empty_with_gpu(
                            gpu.clone(), width, height,
                        )?;
                    creator_shared.set_renderer(renderer);
                    creator_shared.set_shared_gpu(gpu);
                    Ok(match document {
                        Some(document) => Box::new(trd_gui::video_editing::VideoEditingApp::new(
                            document,
                            creator_shared,
                        )),
                        None => Box::new(trd_gui::video_editing::VideoEditingApp::player(
                            player_timeline(width, height),
                            creator_shared,
                        )),
                    })
                }),
            )
            .await?;
        Ok(handle)
    }

    // One or more meshes (repeated `?mesh=`), each an object in the scene. Rust
    // sniffs GLB's `glTF` magic; every other payload is parsed as UTF-8 OBJ.
    let mut loaded: Vec<LoadedMesh> = if mesh_bytes.length() == 0 {
        vec![LoadedMesh {
            mesh: trd_gui::assets::default_mesh()
                .map_err(trd_gui::error::GuiError::from)
                .map_err(to_js)?,
            material: trd_core::DisneyMaterial::default(),
            texture: None,
            metallic_roughness: None,
            normal: None,
            is_gltf: false,
        }]
    } else {
        mesh_bytes
            .iter()
            .map(|value| {
                let bytes = js_sys::Uint8Array::new(&value).to_vec();
                if bytes.starts_with(b"glTF") {
                    let asset = trd_core::import_glb(&bytes)
                        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
                    Ok(LoadedMesh {
                        mesh: asset.mesh,
                        material: asset.material,
                        texture: asset.base_color_texture,
                        metallic_roughness: asset.metallic_roughness_texture,
                        normal: asset.normal_texture,
                        is_gltf: true,
                    })
                } else {
                    let text = String::from_utf8(bytes)
                        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
                    Ok(LoadedMesh {
                        mesh: trd_core::Mesh::from_obj(&text)
                            .map_err(trd_gui::error::GuiError::from)
                            .map_err(to_js)?,
                        material: trd_core::DisneyMaterial::default(),
                        texture: None,
                        metallic_roughness: None,
                        normal: None,
                        is_gltf: false,
                    })
                }
            })
            .collect::<Result<_, wasm_bindgen::JsValue>>()?
    };
    let has_gltf = loaded.iter().any(|asset| asset.is_gltf);
    // One optional albedo texture per mesh (positional: entry `i` skins mesh `i`);
    // an empty/absent entry leaves that object untextured (1×1 white). The JS
    // bootstrap passes an array of `Uint8Array` (one per mesh). Decoded in Rust so
    // trd-core stays I/O-free.
    for (i, asset) in loaded.iter_mut().enumerate() {
        let entry = texture_bytes.get(i as u32);
        let bytes: Vec<u8> = if entry.is_undefined() || entry.is_null() {
            Vec::new()
        } else {
            js_sys::Uint8Array::new(&entry).to_vec()
        };
        if !bytes.is_empty() {
            asset.texture = Some(trd_gui::assets::decode_texture(&bytes).map_err(to_js)?);
        }
    }
    let meshes: Vec<trd_core::Mesh> = loaded.iter().map(|asset| asset.mesh.clone()).collect();
    let textures: Vec<Option<trd_core::ImageTexture>> =
        loaded.iter().map(|asset| asset.texture.clone()).collect();
    let material_maps: Vec<_> = loaded
        .iter()
        .map(|asset| (asset.metallic_roughness.clone(), asset.normal.clone()))
        .collect();
    // The optional HDR env probe (browser `?env=`). Decoded in Rust so trd-core
    // stays I/O-free; when present, the viewer starts in PBR mode.
    let env = match env_bytes {
        Some(bytes) => Some(trd_gui::assets::decode_env_hdr(&bytes).map_err(to_js)?),
        None => None,
    };
    // Per-object mode: start every object in PBR when an env probe is supplied
    // (`?env=`), else Filled — each object's mode is then editable when selected.
    let initial_mode = if env.is_some() || has_gltf {
        trd_core::RenderMode::Shaded
    } else {
        trd_core::RenderMode::Filled
    };
    let lighting = if has_gltf && env.is_some() {
        trd_core::Lighting {
            ambient: 0.0,
            scale: 0.0,
            ..trd_core::Lighting::default()
        }
    } else {
        trd_core::Lighting::default()
    };
    let tone_mapping = if has_gltf {
        trd_core::ToneMapping {
            operator: trd_core::Tonemap::Aces,
            exposure: 1.0,
        }
    } else {
        trd_core::ToneMapping::default()
    };
    // One transform + mode + material per loaded mesh, so `draws()` lays them out
    // side-by-side and each object has its **own** editable render mode + PBR
    // material (#141). `seeded` keeps those per-object vectors the same length.
    let scene = SceneState::seeded(SceneSeed {
        materials: loaded.iter().map(|asset| asset.material.clone()).collect(),
        mode: initial_mode,
        image_based_lighting: trd_core::ImageBasedLighting::default(),
        tone_mapping,
        lighting,
        environment_available: env.is_some(),
    });
    // Render at a resolution suitable for the browser: the canvas's CSS size ×
    // the device pixel ratio, so the image is crisp on high-DPI / large displays
    // instead of upscaling a small fixed buffer. Bounded (aspect-preserving) to
    // keep GPU + readback cost in check.
    let (render_w, render_h) = browser_render_size(&canvas);
    let textures: Vec<Option<&dyn trd_core::Texture>> = textures
        .iter()
        .map(|t| t.as_ref().map(|t| t as &dyn trd_core::Texture))
        .collect();
    let material_maps: Vec<MaterialMaps<'_>> = material_maps
        .iter()
        .map(|(metallic_roughness, normal)| MaterialMaps {
            metallic_roughness: metallic_roughness
                .as_ref()
                .map(|t| t as &dyn trd_core::Texture),
            normal: normal.as_ref().map(|t| t as &dyn trd_core::Texture),
        })
        .collect();
    let renderer = GuiRenderer::new(&meshes, &textures, &material_maps, env, render_w, render_h)
        .await
        .map_err(to_js)?;
    let app = WebApp::new(InteractionController::new(scene), renderer);

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        )
        .await
}

/// A render resolution suitable for the browser: the canvas's CSS size × the
/// device pixel ratio (so the image is crisp on high-DPI / large displays rather
/// than an upscaled small buffer), with the larger axis bounded to [`MAX_DIM`]
/// aspect-preserving to keep GPU + readback cost in check. Falls back to a
/// reasonable size if the canvas isn't laid out yet.
fn browser_render_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    /// Upper bound per axis (aspect-preserving) — crisp yet safe on any GPU.
    const MAX_DIM: f64 = 2048.0;
    const MIN_DIM: u32 = 64;

    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .filter(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(1.0);
    // CSS pixel size from layout (the canvas fills the viewport). Fall back to a
    // reasonable default if it hasn't been laid out yet.
    let (css_w, css_h) = match (canvas.client_width(), canvas.client_height()) {
        (w, h) if w > 1 && h > 1 => (w as f64, h as f64),
        _ => (1280.0, 720.0),
    };
    let (mut w, mut h) = (css_w * dpr, css_h * dpr);
    // Cap the larger axis, scaling both by the same factor to preserve aspect
    // (an off-aspect render would distort the camera and letterbox the display).
    let scale = (MAX_DIM / w.max(h)).min(1.0);
    w *= scale;
    h *= scale;
    let px = |v: f64| (v.round() as u32).max(MIN_DIM);
    (px(w), px(h))
}

/// The placeholder timeline a document-less editor starts on, until the shell
/// has probed the container (#264).
fn player_timeline(width: u32, height: u32) -> trd_core::VideoInfo {
    trd_core::VideoInfo {
        source_name: String::new(),
        mime: String::new(),
        codec: String::new(),
        sha256: String::new(),
        byte_length: 0,
        width,
        height,
        fps_num: 25,
        fps_den: 1,
        frame_count: 1,
        duration_us: 0,
        unpresented_tail_samples: 0,
    }
}

/// The timeline facts the browser bridge needs for frame↔time mapping.///
/// `Copy` in a `Cell` because they are **replaced** when the container is
/// probed: the document's numbers are a starting point, not the truth, and with
/// no document there is nothing but the container (#264).
#[derive(Debug, Clone, Copy)]
struct TimelineFacts {
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    width: u32,
    height: u32,
}

/// Browser bridge for the dedicated editor. It transfers browser-decoded pixels
/// and services commands emitted by Rust UI; it never computes scene matrices.
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct VideoEditingHandle {
    shared: Rc<trd_gui::video_editing::VideoEditingShared>,
    /// The identity a **document** declared, which an opened file must match.
    /// `None` when there is no document — then nothing is expected, so nothing
    /// can mismatch.
    expected: Option<(String, u64)>,
    timeline: std::cell::Cell<TimelineFacts>,
}

impl VideoEditingHandle {
    pub(crate) fn new(
        document: &trd_core::VideoEditingDocument,
        shared: Rc<trd_gui::video_editing::VideoEditingShared>,
    ) -> Self {
        Self {
            shared,
            expected: Some((
                document.video.source_name.clone(),
                document.video.byte_length,
            )),
            timeline: std::cell::Cell::new(TimelineFacts {
                fps_num: document.video.fps_num,
                fps_den: document.video.fps_den,
                frame_count: document.video.frame_count,
                width: document.video.width,
                height: document.video.height,
            }),
        }
    }

    /// The handle for a **plain player**: no document, so no expectations and a
    /// placeholder timeline until the container is probed.
    pub(crate) fn player(shared: Rc<trd_gui::video_editing::VideoEditingShared>) -> Self {
        Self {
            shared,
            expected: None,
            timeline: std::cell::Cell::new(TimelineFacts {
                fps_num: 25,
                fps_den: 1,
                frame_count: 1,
                width: 16,
                height: 9,
            }),
        }
    }
}

#[wasm_bindgen::prelude::wasm_bindgen]
impl VideoEditingHandle {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoFile)]
    pub fn validate_video_file(
        &self,
        filename: &str,
        byte_length: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        // No document, no expectation: any file is a legitimate video to play.
        let Some((expected_name, expected_bytes)) = self.expected.as_ref() else {
            return Ok(());
        };
        if filename != expected_name {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_name}, got {filename}"
            )));
        }
        if byte_length != *expected_bytes as f64 {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_bytes} bytes, got {byte_length:.0}"
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoMetadata)]
    pub fn validate_video_metadata(
        &self,
        width: u32,
        height: u32,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.shared
            .set_video_metadata_observation(width, height, duration_seconds);
        if self.expected.is_none() {
            // Video-first: the container defines the timeline, so its own
            // dimensions and duration cannot disagree with anything (#264).
            return Ok(());
        }
        let timeline = self.timeline.get();
        if (width, height) != (timeline.width, timeline.height) {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}x{} video, got {width}x{height}",
                timeline.width, timeline.height
            )));
        }
        let expected_duration = f64::from(timeline.frame_count) * f64::from(timeline.fps_den)
            / f64::from(timeline.fps_num);
        let frame_duration = f64::from(timeline.fps_den) / f64::from(timeline.fps_num);
        if !duration_seconds.is_finite()
            || (duration_seconds - expected_duration).abs() > frame_duration
        {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_duration:.3}s video, got {duration_seconds:.3}s"
            )));
        }
        Ok(())
    }

    /// Adopts the timeline from a `moov` box the shell located with range reads.
    ///
    /// `<video>` never exposes a frame rate, so without this the browser numbers
    /// frames on an invented grid — a 25 fps clip reported 300 frames instead of
    /// 250. The editor picks the new timeline up on its next frame (#264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoTimelineFromMoov)]
    pub fn set_video_timeline_from_moov(
        &self,
        moov: Vec<u8>,
        source_name: String,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let probed = trd_core::probe_moov(&moov)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("moov has no readable video track"))?;
        self.timeline.set(TimelineFacts {
            fps_num: probed.fps_num,
            fps_den: probed.fps_den,
            frame_count: probed.frame_count,
            width: probed.width,
            height: probed.height,
        });
        self.shared
            .set_pending_video_info(trd_core::VideoInfo::from_probe(probed, source_name));
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = frameIndexAtMediaTime)]
    pub fn frame_index_at_media_time(&self, media_time_seconds: f64) -> u32 {
        let timeline = self.timeline.get();
        trd_gui::video_editing::frame_index_at_media_time(
            media_time_seconds,
            timeline.fps_num,
            timeline.fps_den,
            timeline.frame_count,
        )
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = mediaTimeAtFrame)]
    pub fn media_time_at_frame(&self, frame_index: u32) -> f64 {
        let timeline = self.timeline.get();
        trd_gui::video_editing::media_time_at_frame(
            frame_index,
            timeline.fps_num,
            timeline.fps_den,
            timeline.frame_count,
        )
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = updateVideoFrameRgba)]
    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if frame_index >= self.timeline.get().frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared
            .update_video_frame_rgba(
                rgba,
                width,
                height,
                frame_index,
                media_time_seconds,
                duration_seconds,
            )
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))
    }

    /// Presents a decoded frame without copying it anywhere: the browser
    /// decoded it into GPU memory and it stays there (#229, #282).
    ///
    /// **Takes ownership of the frame** — do not `close()` it in JS. It holds a
    /// slot in a small decoder-side pool, and Rust releases that slot when a
    /// newer frame supersedes this one, *not* once the GPU copy is done: a
    /// render can run more than once for the same frame, since any UI change
    /// repaints, so a frame released after its first upload would leave the
    /// repaint with nothing to draw.
    ///
    /// A separate entry point from
    /// [`update_video_frame_rgba`](Self::update_video_frame_rgba) rather than a
    /// flag, because the preconditions differ — this one carries no buffer at
    /// all.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = presentVideoFrame)]
    pub fn present_video_frame(
        &self,
        frame: web_sys::VideoFrame,
        frame_index: u32,
        media_time_seconds: f64,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        // Wrapped before the first early return, so from here on *every* path
        // releases the decoder-pool slot by dropping — including the rejection
        // below, which used to be one of three hand-written `close()` calls
        // (#302).
        let frame = std::rc::Rc::new(crate::BrowserVideoFrame::new(frame));
        if frame_index >= self.timeline.get().frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared
            .present_external_frame(frame, frame_index, media_time_seconds, duration_seconds)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoStatus)]
    pub fn set_video_status(&self, loaded: bool, playing: bool) {
        self.shared.set_video_status(loaded, playing);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoSourceInfo)]
    pub fn set_video_source_info(
        &self,
        source_kind: u8,
        name: String,
        byte_length: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let kind = match source_kind {
            1 => trd_gui::video_editing::VideoSourceKind::LocalFile,
            2 => trd_gui::video_editing::VideoSourceKind::HttpUrl,
            _ => return Err(wasm_bindgen::JsValue::from_str("unknown video source kind")),
        };
        let byte_length = (byte_length >= 0.0).then_some(byte_length as u64);
        self.shared
            .set_video_source_observation(kind, name, byte_length);
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoMediaState)]
    pub fn set_video_media_state(&self, ready_state: u8, ended: bool) {
        self.shared.set_video_media_observation(ready_state, ended);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoError)]
    pub fn set_video_error(&self, message: String) {
        self.shared.set_error(message);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeCommand)]
    pub fn take_command(&self) -> u8 {
        self.shared.take_command_code()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeAssetRequest)]
    pub fn take_asset_request(&self) -> u8 {
        self.shared.take_asset_request_code()
    }

    /// Loads an annotation document from bytes the shell fetched — a local file
    /// or an HTTP(S) URL, decided by the shell (#264).
    ///
    /// Decoding happens in Rust so native and web share one contract and one
    /// error message; a failure leaves the current document in place.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadDocument)]
    pub fn load_document(&self, bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
        self.shared
            .load_document_bytes(&bytes)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))
    }

    /// Drops the current annotation document: the video keeps playing, as plain
    /// video.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = clearDocument)]
    pub fn clear_document(&self) {
        self.shared.clear_document();
    }

    /// Records what the shell's file picker returned, **without loading it**:
    /// the dialog stays open so an optional document can be chosen too, and its
    /// Load button commits both (#264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setPendingVideoSelection)]
    pub fn set_pending_video_selection(&self, name: String) {
        self.shared
            .set_pending_video(Some(trd_gui::video_editing::PendingSource {
                kind: trd_gui::video_editing::VideoSourceKind::LocalFile,
                name,
            }));
    }

    /// Records the local annotation document the shell's file picker returned.
    /// **Mock**: nothing is decoded yet (#264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setPendingDocumentSelection)]
    pub fn set_pending_document_selection(&self, name: String) {
        self.shared
            .set_pending_document(Some(trd_gui::video_editing::PendingSource {
                kind: trd_gui::video_editing::VideoSourceKind::LocalFile,
                name,
            }));
    }

    /// The pending document's URL, or `None` when the selection is a local file
    /// the shell already holds (or nothing is selected).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = pendingDocumentUrl)]
    pub fn pending_document_url(&self) -> Option<String> {
        self.shared.pending_document().and_then(|source| {
            matches!(
                source.kind,
                trd_gui::video_editing::VideoSourceKind::HttpUrl
            )
            .then_some(source.name)
        })
    }

    /// Whether the dialog has any document selected at all — the shell needs to
    /// know, because Load with none means "play unannotated".
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = hasPendingDocument)]
    pub fn has_pending_document(&self) -> bool {
        self.shared.pending_document().is_some()
    }

    /// The pending video's URL, or `None` when the selection is a local file the
    /// shell already holds.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = pendingVideoUrl)]
    pub fn pending_video_url(&self) -> Option<String> {
        self.shared.pending_video().and_then(|source| {
            matches!(
                source.kind,
                trd_gui::video_editing::VideoSourceKind::HttpUrl
            )
            .then_some(source.name)
        })
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeSeekFrame)]
    pub fn take_seek_frame(&self) -> i32 {
        self.shared.take_seek_frame_code()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadCatalogAsset)]
    pub async fn load_catalog_asset(
        &self,
        asset_code: u8,
        model_bytes: Vec<u8>,
        texture_bytes: Vec<u8>,
        env_bytes: Vec<u8>,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let asset = trd_gui::video_editing::CatalogAsset::from_code(asset_code)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("unknown catalog asset"))?;
        // A catalog swap rebuilds the renderer. It must land on the **same**
        // device as the one egui samples, or the newly registered texture comes
        // from a device the toolkit knows nothing about.
        let renderer = match self.shared.shared_gpu() {
            Some(gpu) => trd_gui::video_editing_renderer::VideoPlacementRenderer::new_with_gpu(
                gpu,
                asset,
                &model_bytes,
                &texture_bytes,
                &env_bytes,
                self.timeline.get().width,
                self.timeline.get().height,
            ),
            None => {
                trd_gui::video_editing_renderer::VideoPlacementRenderer::new(
                    asset,
                    &model_bytes,
                    &texture_bytes,
                    &env_bytes,
                    self.timeline.get().width,
                    self.timeline.get().height,
                )
                .await
            }
        }
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        self.shared.set_catalog_renderer(asset, renderer);
        Ok(())
    }
}
