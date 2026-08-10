//! [`TrdGuiApp`] — the egui application: the display surface and the control
//! panel that together close the interaction loop (#97).
//!
//! The central panel shows the latest RGBA frame (a `trd-core` render uploaded
//! as an egui texture) and is itself the interaction surface: pointer drags and
//! the scroll wheel become normalized [`InteractionEvent`]s, the
//! [`InteractionController`] turns them into a new camera / model matrix, and the
//! scene is re-rendered — the "input → matrix → render → display" cycle. The side
//! panel exposes the render mode, overlay toggles, the primary-drag target, and a
//! reset. This crate holds **no rendering logic**; every pixel comes from
//! `trd-core` via the [`SceneRenderer`].

use std::time::Instant;

use egui::{TextureHandle, TextureOptions};

use trd_gui::interaction::InteractionController;
use trd_gui::render_backend::SceneRenderer;
use trd_gui::ui;

/// The interactive viewer application.
pub struct TrdGuiApp {
    /// Owns the scene and applies interaction gestures to it.
    controller: InteractionController,
    /// The (in-process) backend that renders the scene to RGBA.
    renderer: Box<dyn SceneRenderer>,
    /// The GPU texture the current frame is uploaded into (lazily created).
    texture: Option<TextureHandle>,
    /// Set when the scene changed and the frame must be re-rendered.
    needs_render: bool,
    /// Wall-clock time of the last render, for a simple FPS readout.
    last_render: Option<f32>,
    /// Monotonic clock used to measure render latency.
    clock: Instant,
}

impl TrdGuiApp {
    /// Builds the app around a controller (holding the initial scene) and a
    /// render backend. The first frame renders immediately.
    pub fn new(controller: InteractionController, renderer: Box<dyn SceneRenderer>) -> Self {
        Self {
            controller,
            renderer,
            texture: None,
            needs_render: true,
            last_render: None,
            clock: Instant::now(),
        }
    }

    /// Renders the current scene and (re)uploads it into the display texture.
    /// Logs and keeps the previous frame on error, so a transient render failure
    /// never crashes the UI.
    fn render_scene(&mut self, ctx: &egui::Context) {
        let start = self.clock.elapsed().as_secs_f32();
        match self.renderer.render(&self.controller.state) {
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

        // An expensive backend (e.g. a future Arrow round-trip variant) can ask to
        // re-render only when a pointer interaction *ends*; cheap backends render
        // every changed frame. The first frame (no texture yet) always renders.
        let interacting = ctx.input(|i| i.pointer.any_down());
        let defer = self.renderer.defer_expensive() && interacting;

        if self.texture.is_none() || (self.needs_render && !defer) {
            self.render_scene(&ctx);
        }

        let outcome = ui::show(
            ui,
            &mut ui::View {
                controller: &mut self.controller,
                texture: self.texture.as_ref(),
                render_size: self.renderer.size(),
                last_render_ms: self.last_render,
            },
        );
        self.needs_render |= outcome.needs_render;

        // A click requested a pick: resolve the object under the cursor via the
        // id pass, update the selection, and re-render so its AABB shows (#141).
        if let Some((x, y)) = outcome.pick {
            let hit = self.renderer.pick(&self.controller.state, x, y);
            if hit != self.controller.state.selected {
                self.controller.state.selected = hit;
                self.needs_render = true;
            }
        }

        // While deferring, keep requesting repaints so the pending render fires as
        // soon as the interaction ends.
        if self.needs_render && defer {
            ctx.request_repaint();
        }
    }
}
