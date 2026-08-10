//! trd-gui: the interactive egui front-end (issue #97).
//!
//! trd-gui is a thin **front-end peer** to `trd-app`: it owns the UI, the
//! interaction loop, and scene authoring, and delegates **all** rendering to
//! `trd-core`, honoring the repo invariant that `trd-core` is the single unified
//! rendering core. The loop is:
//!
//! ```text
//! pointer / wheel gesture
//!   → InteractionController (events → camera / model matrix)   [interaction.rs]
//!   → SceneState (orbit camera + object transform)             [scene.rs]
//!   → SceneRenderer::render (trd-core headless RGBA)           [render_backend.rs]
//!   → egui texture in the delivery surface
//! ```
//!
//! It follows **Strategy A** (the decoupled CPU-RGBA handoff): eframe draws the
//! egui UI while `trd-core` renders the scene to an RGBA buffer, so the GUI
//! toolkit stays independent of `trd-core`'s `wgpu 30`. See `docs/gui-design.md`
//! and issue #97 for the full design.
//!
//! ## Module layout
//!
//! `scene`/`interaction`/`ui`/`assets`/`error` are **platform-agnostic** (the
//! scene + controller are unit-tested without egui or a GPU; `ui` is the shared
//! egui layout). The render path is target-split: `native/trd-gui-app` drives the
//! synchronous `render_backend` (`BatchRenderer`); wasm uses the asynchronous
//! offscreen `web_renderer` driven by `web_app`, started via [`start`].

pub mod assets;
pub mod error;
pub mod interaction;
pub mod render_backend;
pub mod scene;
pub mod ui;
pub mod video_editing;
pub mod video_editing_renderer;

#[cfg(target_arch = "wasm32")]
pub mod web_app;
#[cfg(target_arch = "wasm32")]
pub mod web_renderer;

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
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    mesh_bytes: js_sys::Array,
    texture_bytes: js_sys::Array,
    env_bytes: Option<Vec<u8>>,
    backend: Option<String>,
) -> Result<(), wasm_bindgen::JsValue> {
    use crate::interaction::InteractionController;
    use crate::scene::{ObjectTransform, SceneState};
    use crate::web_app::WebApp;
    use crate::web_renderer::{WebBackend, WebRenderer};

    console_error_panic_hook::set_once();
    let _ = eframe::WebLogger::init(log::LevelFilter::Warn);

    let to_js = |e: crate::error::GuiError| wasm_bindgen::JsValue::from_str(&e.to_string());
    struct LoadedMesh {
        mesh: trd_core::Mesh,
        material: trd_core::DisneyMaterial,
        texture: Option<trd_core::ImageTexture>,
        metallic_roughness: Option<trd_core::ImageTexture>,
        normal: Option<trd_core::ImageTexture>,
        is_gltf: bool,
    }

    /// Starts the dedicated `web/gui-video-editing/` poster/document example.
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = startVideoEditing)]
    pub async fn start_video_editing(
        canvas: web_sys::HtmlCanvasElement,
        document_bytes: Vec<u8>,
    ) -> Result<video_editing::VideoEditingHandle, wasm_bindgen::JsValue> {
        use std::rc::Rc;

        console_error_panic_hook::set_once();
        let _ = eframe::WebLogger::init(log::LevelFilter::Warn);
        let document = trd_core::decode_video_editing_document(&document_bytes)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))?;
        let shared = Rc::new(video_editing::VideoEditingShared::default());
        let renderer = video_editing_renderer::VideoPlacementRenderer::new_empty(
            document.video.width,
            document.video.height,
        )
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        shared.set_renderer(renderer);
        let handle = video_editing::VideoEditingHandle::new(&document, shared.clone());
        let app = video_editing::VideoEditingApp::new(document, shared);
        eframe::WebRunner::new()
            .start(
                canvas,
                eframe::WebOptions::default(),
                Box::new(|_cc| Ok(Box::new(app))),
            )
            .await?;
        Ok(handle)
    }

    // One or more meshes (repeated `?mesh=`), each an object in the scene. Rust
    // sniffs GLB's `glTF` magic; every other payload is parsed as UTF-8 OBJ.
    let mut loaded: Vec<LoadedMesh> = if mesh_bytes.length() == 0 {
        vec![LoadedMesh {
            mesh: assets::default_mesh()
                .map_err(crate::error::GuiError::from)
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
                            .map_err(crate::error::GuiError::from)
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
            asset.texture = Some(assets::decode_texture(&bytes).map_err(to_js)?);
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
        Some(bytes) => Some(assets::decode_env_hdr(&bytes).map_err(to_js)?),
        None => None,
    };
    // Per-object mode: start every object in PBR when an env probe is supplied
    // (`?env=`), else Filled — each object's mode is then editable when selected.
    let initial_mode = if env.is_some() || has_gltf {
        trd_core::RenderMode::Pbr
    } else {
        trd_core::RenderMode::Filled
    };
    let lighting = if has_gltf && env.is_some() {
        trd_core::Lighting {
            ambient: 0.0,
            scale: 0.0,
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
    let scene = SceneState {
        // One transform + mode + material per loaded mesh, so `draws()` lays them
        // out side-by-side and each object has its **own** editable render mode +
        // PBR material (#141).
        objects: vec![ObjectTransform::default(); meshes.len()],
        modes: vec![initial_mode; meshes.len()],
        materials: loaded.iter().map(|asset| asset.material.clone()).collect(),
        image_based_lighting: loaded
            .iter()
            .map(|_| trd_core::ImageBasedLighting::default())
            .collect(),
        tone_mappings: vec![tone_mapping; meshes.len()],
        pbr_debug_views: vec![trd_core::PbrDebugView::default(); meshes.len()],
        lighting,
        environment_available: env.is_some(),
        show_environment_background: env.is_some(),
        ..SceneState::default()
    };
    // `?backend=arrow` selects the Arrow wire round-trip; anything else (or
    // absent) is the direct in-process render.
    let backend = match backend.as_deref() {
        Some("arrow") => WebBackend::Arrow,
        _ => WebBackend::Inproc,
    };
    // Render at a resolution suitable for the browser: the canvas's CSS size ×
    // the device pixel ratio, so the image is crisp on high-DPI / large displays
    // instead of upscaling a small fixed buffer. Bounded (aspect-preserving) to
    // keep GPU + readback cost in check.
    let (render_w, render_h) = browser_render_size(&canvas);
    let renderer = WebRenderer::new(
        &meshes,
        &textures,
        &material_maps,
        env,
        render_w,
        render_h,
        backend,
    )
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
#[cfg(target_arch = "wasm32")]
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
