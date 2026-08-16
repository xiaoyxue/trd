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
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = startVideoEditing)]
    pub async fn start_video_editing(
        canvas: web_sys::HtmlCanvasElement,
        document_bytes: Vec<u8>,
    ) -> Result<VideoEditingHandle, wasm_bindgen::JsValue> {
        use std::rc::Rc;

        console_error_panic_hook::set_once();
        let _ = eframe::WebLogger::init(log::LevelFilter::Warn);
        let document = trd_core::decode_video_editing_document(&document_bytes)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
        let shared = Rc::new(trd_gui::video_editing::VideoEditingShared::default());
        let handle = VideoEditingHandle::new(&document, shared.clone());
        let (width, height) = (document.video.width, document.video.height);
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
                    Ok(Box::new(trd_gui::video_editing::VideoEditingApp::new(
                        document,
                        creator_shared,
                    )))
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

/// Browser bridge for the dedicated editor. It transfers browser-decoded pixels
/// and services commands emitted by Rust UI; it never computes scene matrices.
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct VideoEditingHandle {
    shared: Rc<trd_gui::video_editing::VideoEditingShared>,
    source_name: String,
    byte_length: u64,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    width: u32,
    height: u32,
}

impl VideoEditingHandle {
    pub(crate) fn new(
        document: &trd_core::VideoEditingDocument,
        shared: Rc<trd_gui::video_editing::VideoEditingShared>,
    ) -> Self {
        Self {
            shared,
            source_name: document.video.source_name.clone(),
            byte_length: document.video.byte_length,
            fps_num: document.video.fps_num,
            fps_den: document.video.fps_den,
            frame_count: document.video.frame_count,
            width: document.video.width,
            height: document.video.height,
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
        if filename != self.source_name {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}, got {filename}",
                self.source_name
            )));
        }
        if byte_length != self.byte_length as f64 {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {} bytes, got {byte_length:.0}",
                self.byte_length
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
        if (width, height) != (self.width, self.height) {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}x{} video, got {width}x{height}",
                self.width, self.height
            )));
        }
        let expected_duration =
            f64::from(self.frame_count) * f64::from(self.fps_den) / f64::from(self.fps_num);
        let frame_duration = f64::from(self.fps_den) / f64::from(self.fps_num);
        if !duration_seconds.is_finite()
            || (duration_seconds - expected_duration).abs() > frame_duration
        {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_duration:.3}s video, got {duration_seconds:.3}s"
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = frameIndexAtMediaTime)]
    pub fn frame_index_at_media_time(&self, media_time_seconds: f64) -> u32 {
        trd_gui::video_editing::frame_index_at_media_time(
            media_time_seconds,
            self.fps_num,
            self.fps_den,
            self.frame_count,
        )
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = mediaTimeAtFrame)]
    pub fn media_time_at_frame(&self, frame_index: u32) -> f64 {
        trd_gui::video_editing::media_time_at_frame(
            frame_index,
            self.fps_num,
            self.fps_den,
            self.frame_count,
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
    ) -> Result<(), wasm_bindgen::JsValue> {
        if frame_index >= self.frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared
            .update_video_frame_rgba(rgba, width, height, frame_index, media_time_seconds)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))
    }

    /// Hands over the `<video>` element **once**, so later frames present by
    /// index instead of by pixels (#229).
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoElement)]
    pub fn set_video_element(&self, video: web_sys::HtmlVideoElement) {
        self.shared.set_video_element(video);
    }

    /// Presents the element's **current** frame without copying it anywhere: the
    /// browser decoded it into GPU memory and it stays there.
    ///
    /// A separate entry point from
    /// [`update_video_frame_rgba`](Self::update_video_frame_rgba) rather than a
    /// flag, because the preconditions differ — this one requires an element to
    /// have been handed over and carries no buffer at all.
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = presentVideoFrame)]
    pub fn present_video_frame(
        &self,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if frame_index >= self.frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared
            .present_video_frame(width, height, frame_index, media_time_seconds)
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
                self.width,
                self.height,
            ),
            None => {
                trd_gui::video_editing_renderer::VideoPlacementRenderer::new(
                    asset,
                    &model_bytes,
                    &texture_bytes,
                    &env_bytes,
                    self.width,
                    self.height,
                )
                .await
            }
        }
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        self.shared.set_catalog_renderer(asset, renderer);
        Ok(())
    }
}
