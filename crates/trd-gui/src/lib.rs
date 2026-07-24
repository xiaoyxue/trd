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
//! offscreen `wasm_renderer` driven by `web_app`, started via [`start`].

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
pub mod wasm_renderer;
#[cfg(target_arch = "wasm32")]
pub mod web_app;

/// The browser entry point (Slice 4): builds the default-scene offscreen renderer
/// and runs the eframe app on `canvas`. Called from a thin JS bootstrap after the
/// wasm module loads (`await start(canvas)`); all UI + interaction + rendering
/// happen in Rust, per the repo's "JS is a thin bootstrap only" invariant.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), wasm_bindgen::JsValue> {
    use crate::interaction::InteractionController;
    use crate::scene::SceneState;
    use crate::wasm_renderer::WasmRenderer;
    use crate::web_app::WebApp;

    console_error_panic_hook::set_once();
    let _ = eframe::WebLogger::init(log::LevelFilter::Warn);

    let mesh =
        assets::default_mesh().map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let renderer = WasmRenderer::new(&[mesh], None, 512, 512)
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let app = WebApp::new(InteractionController::new(SceneState::default()), renderer);

    eframe::WebRunner::new()
        .start(
            canvas,
            eframe::WebOptions::default(),
            Box::new(|_cc| Ok(Box::new(app))),
        )
        .await
}
