//! [`TrdGuiApp`] — the egui application: the display surface and the control
//! panel that together close the interaction loop (#97).
//!
//! The central panel shows the latest `trd-core` render and is itself the
//! interaction surface: pointer and wheel input become
//! [`InteractionEvent`](trd_gui::interaction::InteractionEvent)s, the
//! [`InteractionController`] turns them into a camera / model matrix, and the
//! scene is re-rendered. This crate holds **no rendering logic**.

use std::time::Instant;

use egui::{TextureHandle, TextureOptions};

use trd_gui::interaction::InteractionController;
use trd_gui::renderer::GuiRenderer;
use trd_gui::ui;

/// The HDR probe a loaded model is lit by when the scene was started without
/// `--env` — the same one the video editor uses for the Dragon.
const DEFAULT_ENV_PATH: &str = "assets/envmap/uffizi-large.hdr";

/// The interactive viewer application.
pub struct TrdGuiApp {
    /// Owns the scene and applies interaction gestures to it.
    controller: InteractionController,
    /// The (in-process) backend that renders the scene to RGBA.
    renderer: GuiRenderer,
    /// The GPU texture the current frame is uploaded into (lazily created).
    texture: Option<TextureHandle>,
    /// Set when the scene changed and the frame must be re-rendered.
    needs_render: bool,
    /// The most recent model-load failure, shown in the panel until a load
    /// succeeds (#353).
    model_error: Option<String>,
    /// The HDR probe bytes a loaded model is lit by when the scene has none —
    /// read once from disk, so repeated loads do not re-read it.
    env_bytes: Option<Vec<u8>>,
    /// Wall-clock time of the last render, for a simple FPS readout.
    last_render: Option<f32>,
    /// Monotonic clock used to measure render latency.
    clock: Instant,
}

impl TrdGuiApp {
    /// Builds the app around a controller (holding the initial scene) and a
    /// render backend. The first frame renders immediately.
    pub fn new(controller: InteractionController, renderer: GuiRenderer) -> Self {
        Self {
            controller,
            renderer,
            texture: None,
            needs_render: true,
            model_error: None,
            env_bytes: None,
            last_render: None,
            clock: Instant::now(),
        }
    }

    /// Opens the file picker and loads the chosen GLB into the live scene.
    ///
    /// A failure is kept for the panel rather than propagated: the point of the
    /// typed error is that a bad file leaves the current scene rendering.
    fn pick_and_load_model(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("glTF binary", &["glb"])
            .set_title("Load model")
            .pick_file()
        else {
            return;
        };
        let name = path.file_name().map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(source) => {
                self.model_error = Some(
                    trd_gui::error::GuiError::MeshIo {
                        path: path.display().to_string(),
                        source,
                    }
                    .to_string(),
                );
                return;
            }
        };
        let request = trd_gui::model::PendingModel {
            name,
            bytes,
            env_bytes: self.default_env_bytes(),
        };
        match trd_gui::model::load_model(&mut self.renderer, &mut self.controller.state, &request) {
            Ok(index) => {
                log::info!("loaded '{}' as object {index}", request.name);
                self.model_error = None;
                self.needs_render = true;
            }
            Err(error) => {
                log::error!("{error}");
                self.model_error = Some(error.to_string());
            }
        }
    }

    /// The built-in Uffizi probe, read once — the same probe the video editor
    /// lights the Dragon with, so an uploaded model matches it (#353).
    fn default_env_bytes(&mut self) -> Option<Vec<u8>> {
        if self.renderer.has_env() {
            return None;
        }
        if self.env_bytes.is_none() {
            match std::fs::read(DEFAULT_ENV_PATH) {
                Ok(bytes) => self.env_bytes = Some(bytes),
                Err(error) => log::error!("failed to read {DEFAULT_ENV_PATH}: {error}"),
            }
        }
        self.env_bytes.clone()
    }

    /// Renders the current scene and (re)uploads it into the display texture.
    /// Logs and keeps the previous frame on error, so a transient render failure
    /// never crashes the UI.
    fn render_scene(&mut self, ctx: &egui::Context) {
        let start = self.clock.elapsed().as_secs_f32();
        // The renderer is async because GPU read-back is; natively the future is
        // already complete when the map poll returns, so blocking is free.
        match pollster::block_on(self.renderer.render(&self.controller.state)) {
            Ok(image) => {
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [image.width as usize, image.height as usize],
                    &image.rgba,
                );
                match &mut self.texture {
                    Some(handle) => handle.set(color, TextureOptions::LINEAR),
                    None => {
                        self.texture =
                            Some(ctx.load_texture("trd-scene", color, TextureOptions::LINEAR));
                    }
                }
                self.last_render = Some(self.clock.elapsed().as_secs_f32() - start);
                self.needs_render = false;
            }
            Err(err) => {
                log::error!("scene render failed: {err}");
                self.needs_render = false;
            }
        }
    }
}

impl eframe::App for TrdGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        // The in-process renderer is cheap enough to run on every changed frame;
        // the first frame (no texture yet) always renders.
        if self.texture.is_none() || self.needs_render {
            self.render_scene(&ctx);
        }

        let outcome = ui::show(
            ui,
            &mut ui::View {
                controller: &mut self.controller,
                texture: self.texture.as_ref(),
                render_size: self.renderer.size(),
                last_render_ms: self.last_render,
                model_error: self.model_error.as_deref(),
            },
        );
        self.needs_render |= outcome.needs_render;

        if outcome.load_model {
            self.pick_and_load_model();
        }

        // A click requested a pick: resolve the object under the cursor via the
        // id pass, update the selection, and re-render so its AABB shows (#141).
        if let Some((x, y)) = outcome.pick {
            let hit = pollster::block_on(self.renderer.pick(&self.controller.state, x, y));
            if hit != self.controller.state.selected {
                self.controller.state.selected = hit;
                self.needs_render = true;
            }
        }
    }
}
