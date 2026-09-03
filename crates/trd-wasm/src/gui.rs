//! The `trd-gui` JS ABI: the interactive viewer (`start`) and the video editor
//! (`startVideoEditing` + [`VideoEditingHandle`]).

use std::rc::Rc;

#[wasm_bindgen::prelude::wasm_bindgen(js_name = videoEditingGltfReferences)]
pub fn video_editing_gltf_references(
    bytes: Vec<u8>,
) -> Result<js_sys::Array, wasm_bindgen::JsValue> {
    let input = trd_gui::video_editing::decode_video_editing_input(&bytes)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
    let result = js_sys::Array::new();
    if let trd_gui::video_editing::VideoEditingInput::Scene(scene) = input {
        for (index, reference) in scene.unresolved_mesh_references() {
            let value = js_sys::Object::new();
            js_sys::Reflect::set(
                &value,
                &wasm_bindgen::JsValue::from_str("index"),
                &wasm_bindgen::JsValue::from_f64(f64::from(index)),
            )?;
            if let Some(path) = reference.path {
                js_sys::Reflect::set(
                    &value,
                    &wasm_bindgen::JsValue::from_str("path"),
                    &wasm_bindgen::JsValue::from_str(&path),
                )?;
            }
            if let Some(url) = reference.url {
                js_sys::Reflect::set(
                    &value,
                    &wasm_bindgen::JsValue::from_str("url"),
                    &wasm_bindgen::JsValue::from_str(&url),
                )?;
            }
            result.push(&value);
        }
    }
    Ok(result)
}

fn resolve_video_editing_scene(
    scene: &mut trd_gui::video_editing::ArrowScene,
    gltf_bytes: &js_sys::Array,
) -> Result<(), wasm_bindgen::JsValue> {
    let references = scene.unresolved_mesh_references();
    if references.len() != gltf_bytes.length() as usize {
        return Err(wasm_bindgen::JsValue::from_str(&format!(
            "expected {} resolved glTF resource(s), got {}",
            references.len(),
            gltf_bytes.length()
        )));
    }
    for ((index, _), bytes) in references.into_iter().zip(gltf_bytes.iter()) {
        scene
            .resolve_gltf(index, &js_sys::Uint8Array::new(&bytes).to_vec())
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
    }
    Ok(())
}

/// Runs the GUI viewer on `canvas`. OBJ/GLB bytes from `?mesh=`, texture from `?texture=`,
/// HDR probe from `?env=` (enables PBR mode); absent parameters fall back to built-in defaults.
///
/// `on_pick_model` is invoked (with no arguments) when the panel's **Load model…**
/// button is pressed, so the shell can open its `<input type=file>`; the chosen
/// bytes come back through [`GuiHandle::load_model`] (#353). The file picker
/// stays in JS because opening one needs a user gesture the browser only grants
/// to the page.
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    mesh_bytes: js_sys::Array,
    texture_bytes: js_sys::Array,
    env_bytes: Option<Vec<u8>>,
    on_pick_model: Option<js_sys::Function>,
) -> Result<GuiHandle, wasm_bindgen::JsValue> {
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

    /// Starts the video editor. Without `document_bytes`, acts as a plain player
    /// with timeline from the container (#264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = startVideoEditing)]
    pub async fn start_video_editing(
        canvas: web_sys::HtmlCanvasElement,
        document_bytes: Option<Vec<u8>>,
        gltf_bytes: js_sys::Array,
        env_bytes: Vec<u8>,
    ) -> Result<VideoEditingHandle, wasm_bindgen::JsValue> {
        use std::rc::Rc;

        console_error_panic_hook::set_once();
        let _ = eframe::WebLogger::init(log::LevelFilter::Warn);
        let input = document_bytes
            .map(|bytes| trd_gui::video_editing::decode_video_editing_input(&bytes))
            .transpose()
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        let (document, scene) = match input {
            Some(trd_gui::video_editing::VideoEditingInput::Annotation(document)) => {
                (Some(document), None)
            }
            Some(trd_gui::video_editing::VideoEditingInput::Scene(mut scene)) => {
                resolve_video_editing_scene(&mut scene, &gltf_bytes)?;
                (None, Some(Rc::new(scene)))
            }
            None => (None, None),
        };
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
        let creator_scene = scene.clone();
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                // Renderer adopts eframe's WebGPU device to avoid a GPU→CPU→GPU round trip per frame.
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
                    let renderer = match creator_scene.as_ref() {
                        Some(scene) => {
                            let assets = scene.mesh_assets()?;
                            trd_gui::video_editing_renderer::VideoPlacementRenderer::
                                new_scene_with_gpu(
                                    gpu.clone(),
                                    &assets,
                                    &env_bytes,
                                    width,
                                    height,
                                )?
                        }
                        None => trd_gui::video_editing_renderer::VideoPlacementRenderer::
                            new_empty_with_gpu(gpu.clone(), width, height)?,
                    };
                    creator_shared.set_renderer(renderer);
                    creator_shared.set_shared_gpu(gpu);
                    Ok(match (document, creator_scene) {
                        (Some(document), _) => Box::new(
                            trd_gui::video_editing::VideoEditingApp::new(document, creator_shared),
                        ),
                        (None, scene) => {
                            let mut app = trd_gui::video_editing::VideoEditingApp::player(
                                player_timeline(width, height),
                                creator_shared,
                            );
                            app.set_arrow_scene(scene);
                            Box::new(app)
                        }
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
    // One optional albedo texture per mesh (positional); absent ⇒ 1×1 white.
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
    let env = match env_bytes {
        Some(bytes) => Some(trd_gui::assets::decode_env_hdr(&bytes).map_err(to_js)?),
        None => None,
    };
    // Start in PBR when a probe or a glTF material is present, else Filled.
    let initial_mode = if env.is_some() || has_gltf {
        trd_core::RenderMode::Shaded
    } else {
        trd_core::RenderMode::Filled
    };
    // The viewer is lit by the probe alone: a key/fill/rim rig on top of image-
    // based lighting double-lights the surface and washes a real PBR material
    // out. `?env=` (or the built-in probe the shell supplies) is what lights it.
    let lighting = trd_gui::scene::ibl_only_lighting();
    let tone_mapping = if has_gltf {
        trd_core::ToneMapping {
            operator: trd_core::Tonemap::Aces,
            exposure: 1.0,
        }
    } else {
        trd_core::ToneMapping::default()
    };
    let scene = SceneState::seeded(SceneSeed {
        materials: loaded.iter().map(|asset| asset.material.clone()).collect(),
        mode: initial_mode,
        image_based_lighting: trd_core::ImageBasedLighting::default(),
        tone_mapping,
        lighting,
        environment_available: env.is_some(),
        // The probe lights the scene; it becomes the backdrop only when the
        // panel's "Environment background" checkbox is ticked.
        show_environment_background: false,
    });
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
    let shared = Rc::new(crate::gui_web_app::GuiShared::new(on_pick_model));
    let app = WebApp::new(InteractionController::new(scene), renderer, shared.clone());

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        )
        .await?;
    Ok(GuiHandle { shared })
}

/// The live viewer, for the shell to push a picked model into (#353).
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct GuiHandle {
    shared: Rc<crate::gui_web_app::GuiShared>,
}

#[wasm_bindgen::prelude::wasm_bindgen]
impl GuiHandle {
    /// Queues a picked GLB to be loaded into the running scene.
    ///
    /// `env_bytes` is the HDR probe to light it by when the viewer was started
    /// without `?env=`; it is ignored once a probe is bound. The load itself
    /// happens on the next frame, when the renderer is not mid-render — nothing
    /// here touches the GPU, so JS never has to know whether a render is in
    /// flight.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadModel)]
    pub fn load_model(&self, name: String, bytes: Vec<u8>, env_bytes: Option<Vec<u8>>) {
        self.shared.queue_model(trd_gui::model::PendingModel {
            name,
            bytes,
            env_bytes,
        });
    }
}

/// CSS size × device pixel ratio, larger axis bounded to `MAX_DIM` (aspect-preserving).
/// Falls back to 1280×720 if not yet laid out.
fn browser_render_size(canvas: &web_sys::HtmlCanvasElement) -> (u32, u32) {
    const MAX_DIM: f64 = 2048.0;
    const MIN_DIM: u32 = 64;

    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .filter(|d| d.is_finite() && *d > 0.0)
        .unwrap_or(1.0);
    let (css_w, css_h) = match (canvas.client_width(), canvas.client_height()) {
        (w, h) if w > 1 && h > 1 => (w as f64, h as f64),
        _ => (1280.0, 720.0),
    };
    let (mut w, mut h) = (css_w * dpr, css_h * dpr);
    let scale = (MAX_DIM / w.max(h)).min(1.0);
    w *= scale;
    h *= scale;
    let px = |v: f64| (v.round() as u32).max(MIN_DIM);
    (px(w), px(h))
}

/// Placeholder timeline until the container is probed (#264).
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
        unpresented_tail: None,
    }
}

/// Timeline facts for frame↔time mapping; replaced when the container is probed (#264).
#[derive(Debug, Clone, Copy)]
struct TimelineFacts {
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    width: u32,
    height: u32,
}

/// Browser bridge for the video editor: transfers decoded frames, services Rust UI commands.
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct VideoEditingHandle {
    shared: Rc<trd_gui::video_editing::VideoEditingShared>,
    /// File identity from the document; `None` when document-less (no check).
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

    /// Plain-player handle: no document, placeholder timeline until container is probed.
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
            // Video-first: container defines the timeline, nothing to check (#264).
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

    /// Adopts rational fps + frame count from a range-read `moov` box.
    /// `<video>` never exposes frame rate directly; the editor picks this up on its next frame (#264).
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

    /// Presents a browser-decoded frame that stays in GPU memory (#229, #282).
    /// **Takes ownership** — do not `close()` it in JS; the decoder-pool slot is
    /// released when a newer frame supersedes it.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = presentVideoFrame)]
    pub fn present_video_frame(
        &self,
        frame: web_sys::VideoFrame,
        frame_index: u32,
        media_time_seconds: f64,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        // Wrap before the bounds check so the rejection path also releases the
        // decoder-pool slot by dropping (#302).
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

    /// Surfaces a scoped failure; a success elsewhere cannot clear it (#329).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setError)]
    pub fn set_error(&self, scope: u8, message: String) -> Result<(), wasm_bindgen::JsValue> {
        let scope = trd_gui::video_editing::ErrorScope::from_code(scope)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("unknown error scope"))?;
        self.shared.set_error(scope, message);
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeCommand)]
    pub fn take_command(&self) -> u8 {
        self.shared.take_command_code()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = pendingArrowExportFilename)]
    pub fn pending_arrow_export_filename(&self) -> Option<String> {
        self.shared.pending_arrow_export_filename()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeExportArrow)]
    pub fn take_export_arrow(&self) -> Result<Vec<u8>, wasm_bindgen::JsValue> {
        self.shared
            .take_arrow_export()
            .map(|export| export.bytes)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("no Arrow export is queued"))
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = finishArrowExport)]
    pub fn finish_arrow_export(&self, success: bool, message: String) {
        self.shared
            .complete_arrow_export(if success { Ok(message) } else { Err(message) });
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = cancelArrowExport)]
    pub fn cancel_arrow_export(&self) {
        self.shared.cancel_arrow_export();
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeAssetRequest)]
    pub fn take_asset_request(&self) -> u8 {
        self.shared.take_asset_request_code()
    }

    /// Loads an annotation document or exported protocol scene from bytes.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadDocument)]
    pub async fn load_document(&self, bytes: Vec<u8>) -> Result<(), wasm_bindgen::JsValue> {
        self.load_document_with_gltf(bytes, js_sys::Array::new(), Vec::new())
            .await
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadDocumentWithGltf)]
    pub async fn load_document_with_gltf(
        &self,
        bytes: Vec<u8>,
        gltf_bytes: js_sys::Array,
        env_bytes: Vec<u8>,
    ) -> Result<(), wasm_bindgen::JsValue> {
        match trd_gui::video_editing::decode_video_editing_input(&bytes)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?
        {
            trd_gui::video_editing::VideoEditingInput::Annotation(document) => {
                self.shared.queue_annotation_document(document);
            }
            trd_gui::video_editing::VideoEditingInput::Scene(mut scene) => {
                resolve_video_editing_scene(&mut scene, &gltf_bytes)?;
                let assets = scene
                    .mesh_assets()
                    .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
                let timeline = self.timeline.get();
                let renderer = match self.shared.shared_gpu() {
                    Some(gpu) => {
                        trd_gui::video_editing_renderer::VideoPlacementRenderer::new_scene_with_gpu(
                            gpu,
                            &assets,
                            &env_bytes,
                            timeline.width,
                            timeline.height,
                        )
                    }
                    None => {
                        trd_gui::video_editing_renderer::VideoPlacementRenderer::new_scene(
                            &assets,
                            &env_bytes,
                            timeline.width,
                            timeline.height,
                        )
                        .await
                    }
                }
                .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
                self.shared.set_renderer(renderer);
                self.shared.queue_arrow_scene(Rc::new(scene));
            }
        }
        Ok(())
    }

    /// Drops the current annotation document; video keeps playing.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = clearDocument)]
    pub fn clear_document(&self) {
        self.shared.clear_document();
    }

    /// Records the pending video selection (loading deferred to Load, #264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setPendingVideoSelection)]
    pub fn set_pending_video_selection(&self, name: String) {
        self.shared
            .set_pending_video(Some(trd_gui::video_editing::PendingSource {
                kind: trd_gui::video_editing::VideoSourceKind::LocalFile,
                name,
            }));
    }

    /// Records the pending document selection (not decoded yet, #264).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setPendingDocumentSelection)]
    pub fn set_pending_document_selection(&self, name: String) {
        self.shared
            .set_pending_document(Some(trd_gui::video_editing::PendingSource {
                kind: trd_gui::video_editing::VideoSourceKind::LocalFile,
                name,
            }));
    }

    /// URL of the pending document; `None` for local-file selections or no selection.
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

    /// `true` if a document is selected (Load with none means play unannotated).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = hasPendingDocument)]
    pub fn has_pending_document(&self) -> bool {
        self.shared.pending_document().is_some()
    }

    /// URL of the pending video; `None` for local-file selections.
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
        source_path: String,
        source_url: String,
        model_bytes: Vec<u8>,
        texture_bytes: Vec<u8>,
        env_bytes: Vec<u8>,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let asset = trd_gui::video_editing::CatalogAsset::from_code(asset_code)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("unknown catalog asset"))?;
        let source = trd_core::MeshReference::new(Some(source_path), Some(source_url))
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("catalog asset reference is empty"))?;
        // Must use the same device egui samples — a different device yields an unusable texture.
        let renderer = match self.shared.shared_gpu() {
            Some(gpu) => trd_gui::video_editing_renderer::VideoPlacementRenderer::new_with_gpu(
                gpu,
                asset,
                source.clone(),
                &model_bytes,
                &texture_bytes,
                &env_bytes,
                self.timeline.get().width,
                self.timeline.get().height,
            ),
            None => {
                trd_gui::video_editing_renderer::VideoPlacementRenderer::new(
                    asset,
                    source,
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
        self.shared
            .clear_error(trd_gui::video_editing::ErrorScope::Catalog);
        Ok(())
    }
}
