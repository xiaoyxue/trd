//! Shared egui UI for the interactive viewer (#97) — the side control panel and
//! the central image surface, written **once** and used by both the native
//! window ([`crate::app`]) and the wasm/browser app ([`crate::web_app`]).
//!
//! It is rendering-backend-agnostic: it only reads the current display texture
//! and render size and mutates the [`InteractionController`] from pointer/scroll
//! input, returning whether the scene changed (so the host decides *when* and
//! *how* to re-render — synchronously on native, asynchronously on wasm).

use egui::{Color32, PointerButton, Sense, TextureHandle, Vec2};
use trd_core::{RenderMode, Tonemap};

use crate::interaction::{
    AxisConstraint, InteractionController, InteractionEvent, InteractionTarget, TransformMode,
};
use crate::scene::{MAX_SCALE, MIN_SCALE};

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
        .resizable(true)
        .default_size(264.0)
        .min_size(240.0)
        .max_size(420.0)
        .show(ui, |ui| {
            needs_render |= controls_panel(ui, view);
        });
    egui::CentralPanel::default().show(ui, |ui| {
        needs_render |= image_panel(ui, view);
    });
    needs_render
}

/// The left control panel: primary-drag target, render mode, overlays, reset, and
/// a status readout. Grouped into collapsible sections inside a scroll area so the
/// (now taller) panel stays usable. Returns whether the scene changed.
fn controls_panel(ui: &mut egui::Ui, view: &mut View) -> bool {
    let mut needs_render = false;
    ui.horizontal(|ui| {
        ui.heading("trd-gui");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(format!("proto {}", trd_core::PROTOCOL_VERSION));
        });
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        // ── Interaction ──────────────────────────────────────────────────
        section(ui, "Interaction", |ui| {
            let mut c = false;
            ui.label("Primary drag");
            let target = &mut view.controller.target;
            ui.horizontal_wrapped(|ui| {
                c |= ui
                    .selectable_value(target, InteractionTarget::Camera, "Orbit camera")
                    .changed();
                c |= ui
                    .selectable_value(target, InteractionTarget::Object, "Object")
                    .changed();
            });
            // When a primary drag targets the object, pick which transform it
            // edits, and (for rotate/move) optionally lock it to one axis.
            if view.controller.target == InteractionTarget::Object {
                ui.add_space(4.0);
                ui.label("Manipulate");
                let mode = &mut view.controller.mode;
                ui.horizontal_wrapped(|ui| {
                    ui.selectable_value(mode, TransformMode::Rotate, "Rotate");
                    ui.selectable_value(mode, TransformMode::Move, "Move");
                    ui.selectable_value(mode, TransformMode::Scale, "Scale");
                });
                if matches!(
                    view.controller.mode,
                    TransformMode::Rotate | TransformMode::Move
                ) {
                    ui.add_space(4.0);
                    ui.label("Axis lock");
                    let axis = &mut view.controller.axis;
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(axis, AxisConstraint::Free, "Free");
                        ui.selectable_value(axis, AxisConstraint::X, "X");
                        ui.selectable_value(axis, AxisConstraint::Y, "Y");
                        ui.selectable_value(axis, AxisConstraint::Z, "Z");
                    });
                }
            }
            c
        });

        // ── Transform (numeric widgets, in sync with the mouse) ───────────
        section(ui, "Transform", |ui| transform_panel(ui, view));

        // ── Render mode ──────────────────────────────────────────────────
        section(ui, "Render mode", |ui| {
            let mut c = false;
            let mode = &mut view.controller.state.mode;
            ui.horizontal_wrapped(|ui| {
                c |= ui
                    .selectable_value(mode, RenderMode::Filled, "Filled")
                    .changed();
                c |= ui
                    .selectable_value(mode, RenderMode::Wireframe, "Wireframe")
                    .changed();
                c |= ui
                    .selectable_value(mode, RenderMode::Textured, "Textured")
                    .changed();
                c |= ui.selectable_value(mode, RenderMode::Pbr, "PBR").changed();
            });
            c
        });

        // The Disney PBR material controls, only while PBR mode is selected.
        if view.controller.state.mode == RenderMode::Pbr {
            section(ui, "PBR material", |ui| pbr_panel(ui, view));
        }

        // ── Overlays ─────────────────────────────────────────────────────
        section(ui, "Overlays", |ui| {
            let mut c = false;
            let state = &mut view.controller.state;
            ui.label("Gizmos");
            c |= ui.checkbox(&mut state.show_aabb, "Bounding box").changed();
            c |= ui.checkbox(&mut state.show_axes, "World axes").changed();
            c |= ui.checkbox(&mut state.show_local_axes, "Local axes").changed();
            ui.add_space(4.0);
            ui.label("Plane grid (XZ)");
            c |= ui.checkbox(&mut state.show_world_grid, "World grid").changed();
            c |= ui.checkbox(&mut state.show_local_grid, "Local grid").changed();
            c
        });

        ui.add_space(4.0);
        if ui.button("Reset view").clicked() {
            needs_render |= view.controller.apply(InteractionEvent::Reset);
        }
        ui.separator();

        // ── Status ───────────────────────────────────────────────────────
        if let Some(ms) = view.last_render_ms {
            ui.weak(format!("Last render: {:.1} ms", ms * 1000.0));
        }
        let (w, h) = view.render_size;
        ui.weak(format!("Render size: {w}×{h}"));
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(
                "Left-drag: orbit / rotate / move / scale\nAxis lock: drag rotates/moves on one axis\nRight-drag: move object\nScroll: zoom (or scale)",
            )
            .small()
            .color(Color32::GRAY),
        );
    });
    needs_render
}

/// A collapsible, default-open control section with a bold header. Runs `body`
/// (which reports whether the scene changed) and folds its result back out, so
/// the caller's `needs_render` accounting is unchanged by the grouping.
fn section(ui: &mut egui::Ui, title: &str, body: impl FnOnce(&mut egui::Ui) -> bool) -> bool {
    egui::CollapsingHeader::new(egui::RichText::new(title).strong())
        .default_open(true)
        .show(ui, body)
        .body_returned
        .unwrap_or(false)
}

/// The object **transform** widgets: numeric translation (x/y/z), rotation
/// (yaw/pitch/roll in degrees), and scale (uniform + per-axis). These edit the
/// exact same [`ObjectTransform`](crate::scene::ObjectTransform) the mouse
/// gestures mutate, so the two input paths stay numerically in sync — dragging in
/// the image updates these fields, and editing a field moves the object. Returns
/// whether the transform changed.
fn transform_panel(ui: &mut egui::Ui, view: &mut View) -> bool {
    let mut changed = false;
    let obj = &mut view.controller.state.object;
    ui.label("Transform");

    ui.label("Translation");
    ui.horizontal(|ui| {
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            ui.label(*axis);
            changed |= ui
                .add(egui::DragValue::new(&mut obj.translation[i]).speed(0.01))
                .changed();
        }
    });

    ui.label("Rotation (°)");
    changed |= angle_row(ui, "X (pitch)", &mut obj.pitch);
    changed |= angle_row(ui, "Y (yaw)", &mut obj.yaw);
    changed |= angle_row(ui, "Z (roll)", &mut obj.roll);

    ui.label("Scale");
    // Uniform scale drives all three axes together; per-axis rows allow a
    // non-uniform scale. Both clamp to the object's working range.
    let mut uniform = (obj.scale[0] + obj.scale[1] + obj.scale[2]) / 3.0;
    ui.horizontal(|ui| {
        ui.label("Uniform");
        if ui
            .add(
                egui::DragValue::new(&mut uniform)
                    .speed(0.01)
                    .range(MIN_SCALE..=MAX_SCALE),
            )
            .changed()
        {
            obj.scale = [uniform, uniform, uniform];
            changed = true;
        }
    });
    ui.horizontal(|ui| {
        for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
            ui.label(*axis);
            changed |= ui
                .add(
                    egui::DragValue::new(&mut obj.scale[i])
                        .speed(0.01)
                        .range(MIN_SCALE..=MAX_SCALE),
                )
                .changed();
        }
    });
    changed
}

/// A single labeled angle row that displays/edits `radians` in **degrees** (the
/// natural unit for the widget), storing back radians. Returns whether it changed.
fn angle_row(ui: &mut egui::Ui, label: &str, radians: &mut f32) -> bool {
    let mut deg = radians.to_degrees();
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(label);
        if ui
            .add(egui::DragValue::new(&mut deg).speed(1.0).suffix("°"))
            .changed()
        {
            *radians = deg.to_radians();
            changed = true;
        }
    });
    changed
}

/// The Disney PBR material sub-panel (shown only in [`RenderMode::Pbr`]): live
/// sliders for the parameters that most change the look — metallic, roughness,
/// environment-reflection gain, and tone-map exposure — plus a Reinhard/ACES
/// tone-map selector. Editing any of them re-renders the scene, so the material
/// is as interactive as the camera. Returns whether the material changed.
fn pbr_panel(ui: &mut egui::Ui, view: &mut View) -> bool {
    let mut changed = false;
    let pbr = &mut view.controller.state.pbr;
    ui.label("PBR material");
    // Label each slider on its own line so the text never clips in a narrow
    // panel; the slider then spans the full panel width beneath it.
    ui.label("Metallic");
    changed |= ui
        .add(egui::Slider::new(&mut pbr.metallic, 0.0..=1.0))
        .changed();
    ui.label("Roughness");
    changed |= ui
        .add(egui::Slider::new(&mut pbr.roughness, 0.0..=1.0))
        .changed();
    ui.label("Clearcoat");
    changed |= ui
        .add(egui::Slider::new(&mut pbr.clearcoat, 0.0..=1.0))
        .changed();
    ui.label("Env intensity");
    changed |= ui
        .add(egui::Slider::new(&mut pbr.env_intensity, 0.0..=4.0))
        .changed();
    ui.label("Exposure");
    changed |= ui
        .add(egui::Slider::new(&mut pbr.exposure, 0.0..=4.0))
        .changed();
    ui.label("Tonemap");
    ui.horizontal_wrapped(|ui| {
        changed |= ui
            .selectable_value(&mut pbr.tonemap, Tonemap::Reinhard, "Reinhard")
            .changed();
        changed |= ui
            .selectable_value(&mut pbr.tonemap, Tonemap::Aces, "ACES")
            .changed();
    });
    changed
}
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
            let delta = scroll / 100.0;
            // In object Scale mode the wheel scales the object; otherwise it
            // dollies the camera (the default zoom).
            let event = if view.controller.target == InteractionTarget::Object
                && view.controller.mode == TransformMode::Scale
            {
                InteractionEvent::Scale { delta }
            } else {
                InteractionEvent::Zoom { delta }
            };
            changed |= view.controller.apply(event);
        }
    }

    if changed {
        ui.ctx().request_repaint();
    }
    changed
}
