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

use egui::{Color32, PointerButton, Sense, TextureHandle, TextureOptions, Vec2};
use trd_core::RenderMode;

use crate::interaction::{InteractionController, InteractionEvent, InteractionTarget};
use crate::render_backend::SceneRenderer;

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

    /// The left control panel: primary-drag target, render mode, overlays, reset,
    /// and a status readout. Returns nothing; it mutates the scene directly.
    fn controls_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("trd-gui");
        ui.label(format!("trd-core protocol {}", trd_core::PROTOCOL_VERSION));
        ui.separator();

        ui.label("Primary drag");
        let target = &mut self.controller.target;
        ui.horizontal(|ui| {
            self.needs_render |= ui
                .selectable_value(target, InteractionTarget::Camera, "Orbit camera")
                .changed();
            self.needs_render |= ui
                .selectable_value(target, InteractionTarget::Object, "Rotate object")
                .changed();
        });
        ui.separator();

        ui.label("Render mode");
        let mode = &mut self.controller.state.mode;
        ui.horizontal(|ui| {
            self.needs_render |= ui
                .selectable_value(mode, RenderMode::Filled, "Filled")
                .changed();
            self.needs_render |= ui
                .selectable_value(mode, RenderMode::Wireframe, "Wireframe")
                .changed();
        });
        ui.separator();

        ui.label("Overlays");
        let state = &mut self.controller.state;
        self.needs_render |= ui.checkbox(&mut state.show_aabb, "Bounding box").changed();
        self.needs_render |= ui.checkbox(&mut state.show_axes, "World axes").changed();
        self.needs_render |= ui
            .checkbox(&mut state.show_local_axes, "Local axes")
            .changed();
        ui.separator();

        if ui.button("Reset view").clicked() {
            self.needs_render |= self.controller.apply(InteractionEvent::Reset);
        }
        ui.separator();

        if let Some(ms) = self.last_render {
            ui.label(format!("Last render: {:.1} ms", ms * 1000.0));
        }
        let (w, h) = self.renderer.size();
        ui.label(format!("Render size: {w}×{h}"));
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new("Left-drag: orbit / rotate\nRight-drag: move object\nScroll: zoom")
                .small()
                .color(Color32::GRAY),
        );
    }

    /// The central image panel: shows the current frame scaled to fit and turns
    /// pointer/scroll input over it into interaction events.
    fn image_panel(&mut self, ui: &mut egui::Ui) {
        let Some(texture) = self.texture.clone() else {
            ui.centered_and_justified(|ui| {
                ui.label("Rendering…");
            });
            return;
        };

        let (img_w, img_h) = self.renderer.size();
        let img_aspect = img_w as f32 / img_h.max(1) as f32;
        let avail = ui.available_size();
        // Fit the image inside the panel, preserving aspect (letterboxed).
        let disp = if avail.x / avail.y.max(1.0) > img_aspect {
            Vec2::new(avail.y * img_aspect, avail.y)
        } else {
            Vec2::new(avail.x, avail.x / img_aspect)
        };

        let response = ui.centered_and_justified(|ui| {
            ui.add(
                egui::Image::new(egui::load::SizedTexture::from_handle(&texture))
                    .fit_to_exact_size(disp)
                    .sense(Sense::drag()),
            )
        });
        let response = response.inner;

        let size = response.rect.size();
        if size.x <= 0.0 || size.y <= 0.0 {
            return;
        }
        let delta = response.drag_delta();
        let (dx, dy) = (delta.x / size.x, delta.y / size.y);

        let mut changed = false;
        if response.dragged_by(PointerButton::Primary) {
            changed |= self.controller.apply(InteractionEvent::Primary { dx, dy });
        } else if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
        {
            changed |= self.controller.apply(InteractionEvent::Pan { dx, dy });
        }

        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                changed |= self.controller.apply(InteractionEvent::Zoom {
                    delta: scroll / 100.0,
                });
            }
        }

        if changed {
            self.needs_render = true;
            ui.ctx().request_repaint();
        }
    }
}

impl eframe::App for TrdGuiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        egui::Panel::left("controls")
            .resizable(false)
            .exact_size(200.0)
            .show(ui, |ui| self.controls_panel(ui));

        egui::CentralPanel::default().show(ui, |ui| {
            // Render before displaying so control-panel edits made this frame
            // (mode/overlay toggles set `needs_render`) show immediately.
            if self.needs_render || self.texture.is_none() {
                self.render_scene(&ctx);
            }
            self.image_panel(ui);
        });
    }
}
