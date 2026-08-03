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
//!   → egui texture in the central panel                        [app.rs]
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
//! egui layout). The render path is target-split: native uses the synchronous
//! `render_backend` (`BatchRenderer`) driven by `app`; wasm uses the asynchronous
//! offscreen `web_renderer` driven by `web_app`, started via [`start`].

pub mod assets;
pub mod error;
pub mod interaction;
pub mod render_backend;
pub mod scene;
pub mod ui;

#[cfg(not(target_arch = "wasm32"))]
pub mod app;
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;

#[cfg(target_arch = "wasm32")]
pub mod web_app;
#[cfg(target_arch = "wasm32")]
pub mod web_renderer;

/// The browser entry point (Slice 4): builds the offscreen renderer and runs the
/// eframe app on `canvas`. `mesh_obj`, `texture_bytes`, and `env_bytes` are the
/// browser equivalents of the native `--mesh` / `--texture` / `--env` flags — an
/// optional Wavefront OBJ **as text**, optional texture image **bytes**
/// (PNG/JPEG), and an optional Radiance HDR environment probe **bytes**; the thin
/// JS bootstrap fetches them from `?mesh=` / `?texture=` / `?env=` URLs and passes
/// them in. `None`/absent falls back to the built-in cube / no texture / no probe.
/// Supplying an env probe starts the viewer in Disney **PBR** mode (the material
/// is then editable live in the UI). All UI + interaction + rendering happen in
/// Rust, per the repo's "JS is a thin bootstrap only" invariant.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(
    canvas: web_sys::HtmlCanvasElement,
    mesh_objs: Vec<String>,
    texture_bytes: Option<Vec<u8>>,
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
    // One or more meshes (repeated `?mesh=`), each an object in the scene; an
    // empty list falls back to the built-in cube.
    let meshes: Vec<trd_core::Mesh> = if mesh_objs.is_empty() {
        vec![assets::default_mesh()
            .map_err(crate::error::GuiError::from)
            .map_err(to_js)?]
    } else {
        mesh_objs
            .iter()
            .map(|text| {
                trd_core::Mesh::from_obj(text)
                    .map_err(crate::error::GuiError::from)
                    .map_err(to_js)
            })
            .collect::<Result<_, _>>()?
    };
    let texture = match texture_bytes {
        Some(bytes) => Some(assets::decode_texture(&bytes).map_err(to_js)?),
        None => None,
    };
    // The optional HDR env probe (browser `?env=`). Decoded in Rust so trd-core
    // stays I/O-free; when present, the viewer starts in PBR mode.
    let env = match env_bytes {
        Some(bytes) => Some(assets::decode_env_hdr(&bytes).map_err(to_js)?),
        None => None,
    };
    let scene = SceneState {
        mode: if env.is_some() {
            trd_core::RenderMode::Pbr
        } else {
            trd_core::RenderMode::Filled
        },
        // One transform per loaded mesh, so `draws()` lays them out side-by-side.
        objects: vec![ObjectTransform::default(); meshes.len()],
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
        texture.as_ref().map(|t| t as &dyn trd_core::Texture),
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
