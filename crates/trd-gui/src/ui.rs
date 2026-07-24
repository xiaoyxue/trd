//! Shared egui UI for the interactive viewer (#97) — the side control panel and
//! the central image surface, written **once** and used by both the native
//! window ([`crate::app`]) and the wasm/browser app ([`crate::web_app`]).
//!
//! It is rendering-backend-agnostic: it only reads the current display texture
//! and render size and mutates the [`InteractionController`] from pointer/scroll
//! input, returning whether the scene changed (so the host decides *when* and
//! *how* to re-render — synchronously on native, asynchronously on wasm).

use egui::{Color32, PointerButton, Sense, TextureHandle, Vec2};
use trd_core::RenderMode;

use crate::interaction::{InteractionController, InteractionEvent, InteractionTarget};

/// The per-frame view the shared UI draws: the interaction state, the current
/// display texture (if a frame has been rendered), the render resolution, and an
/// optional last-render time for the status readout.
pub struct View<'a> {
    pub controller: &'a mut InteractionController,
    pub texture: Option<&'a TextureHandle>,
    pub render_size: (u32, u32),
    pub last_render_ms: Option<f32>,
}

/// Lays out the side controls + central image and maps input to interaction
/// events. Returns `true` if the scene changed and a re-render is needed.
pub fn show(ui: &mut egui::Ui, view: &mut View) -> bool {
    let mut needs_render = false;
    egui::Panel::left("controls")
        .resizable(false)
        .exact_size(200.0)
        .show(ui, |ui| {
            needs_render |= controls_panel(ui, view);
        });
    egui::CentralPanel::default().show(ui, |ui| {
        needs_render |= image_panel(ui, view);
    });
    needs_render
}

/// The left control panel: primary-drag target, render mode, overlays, reset, and
/// a status readout. Returns whether the scene changed.
fn controls_panel(ui: &mut egui::Ui, view: &mut View) -> bool {
    let mut needs_render = false;
    ui.heading("trd-gui");
    ui.label(format!("trd-core protocol {}", trd_core::PROTOCOL_VERSION));
    ui.separator();

    ui.label("Primary drag");
    let target = &mut view.controller.target;
    ui.horizontal(|ui| {
        needs_render |= ui
            .selectable_value(target, InteractionTarget::Camera, "Orbit camera")
            .changed();
        needs_render |= ui
            .selectable_value(target, InteractionTarget::Object, "Rotate object")
            .changed();
    });
    ui.separator();

    ui.label("Render mode");
    let mode = &mut view.controller.state.mode;
    ui.horizontal(|ui| {
        needs_render |= ui
            .selectable_value(mode, RenderMode::Filled, "Filled")
            .changed();
        needs_render |= ui
            .selectable_value(mode, RenderMode::Wireframe, "Wireframe")
            .changed();
        needs_render |= ui
            .selectable_value(mode, RenderMode::Textured, "Textured")
            .changed();
    });
    ui.separator();

    ui.label("Overlays");
    let state = &mut view.controller.state;
    needs_render |= ui.checkbox(&mut state.show_aabb, "Bounding box").changed();
    needs_render |= ui.checkbox(&mut state.show_axes, "World axes").changed();
    needs_render |= ui
        .checkbox(&mut state.show_local_axes, "Local axes")
        .changed();
    ui.separator();

    if ui.button("Reset view").clicked() {
        needs_render |= view.controller.apply(InteractionEvent::Reset);
    }
    ui.separator();

    if let Some(ms) = view.last_render_ms {
        ui.label(format!("Last render: {:.1} ms", ms * 1000.0));
    }
    let (w, h) = view.render_size;
    ui.label(format!("Render size: {w}×{h}"));
    ui.add_space(8.0);
    ui.label(
        egui::RichText::new("Left-drag: orbit / rotate\nRight-drag: move object\nScroll: zoom")
            .small()
            .color(Color32::GRAY),
    );
    needs_render
}

/// The central image panel: shows the current frame scaled to fit and turns
/// pointer/scroll input over it into interaction events. Returns whether the
/// scene changed.
fn image_panel(ui: &mut egui::Ui, view: &mut View) -> bool {
    let Some(texture) = view.texture else {
        ui.centered_and_justified(|ui| {
            ui.label("Rendering…");
        });
        return false;
    };

    let (img_w, img_h) = view.render_size;
    let img_aspect = img_w as f32 / img_h.max(1) as f32;
    let avail = ui.available_size();
    // Fit the image inside the panel, preserving aspect (letterboxed).
    let disp = if avail.x / avail.y.max(1.0) > img_aspect {
        Vec2::new(avail.y * img_aspect, avail.y)
    } else {
        Vec2::new(avail.x, avail.x / img_aspect)
    };

    let response = ui
        .centered_and_justified(|ui| {
            ui.add(
                egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                    .fit_to_exact_size(disp)
                    .sense(Sense::drag()),
            )
        })
        .inner;

    let size = response.rect.size();
    if size.x <= 0.0 || size.y <= 0.0 {
        return false;
    }
    let delta = response.drag_delta();
    let (dx, dy) = (delta.x / size.x, delta.y / size.y);

    let mut changed = false;
    if response.dragged_by(PointerButton::Primary) {
        changed |= view.controller.apply(InteractionEvent::Primary { dx, dy });
    } else if response.dragged_by(PointerButton::Secondary)
        || response.dragged_by(PointerButton::Middle)
    {
        changed |= view.controller.apply(InteractionEvent::Pan { dx, dy });
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            changed |= view.controller.apply(InteractionEvent::Zoom {
                delta: scroll / 100.0,
            });
        }
    }

    if changed {
        ui.ctx().request_repaint();
    }
    changed
}
