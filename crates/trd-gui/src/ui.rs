//! Shared egui UI for the interactive viewer (#97) — the side control panel and
//! the central image surface, written **once** and used by both the native
//! window ([`crate::app`]) and the wasm/browser app ([`crate::web_app`]).
//!
//! It is rendering-backend-agnostic: it only reads the current display texture
//! and render size and mutates the [`InteractionController`] from pointer/scroll
//! input, returning whether the scene changed (so the host decides *when* and
//! *how* to re-render — synchronously on native, asynchronously on wasm).

use egui::{Color32, PointerButton, Sense, TextureHandle, Vec2};
use trd_core::{PbrDebugView, RenderMode, Tonemap};

use crate::interaction::{
    AxisConstraint, InteractionController, InteractionEvent, InteractionTarget, MoveDirection,
    TransformMode,
};
use crate::scene::{MAX_SCALE, MIN_SCALE};

pub type ExtraControls<'a> = dyn FnMut(&mut egui::Ui) -> bool + 'a;
pub type ImageOverlay<'a> = dyn FnMut(&egui::Ui, egui::Rect) + 'a;
pub type CentralBottom<'a> = dyn FnMut(&mut egui::Ui) + 'a;

/// How the render target is sized inside the central panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageSizing {
    /// Preserve aspect while fitting the entire image inside the panel.
    #[default]
    FitCanvas,
    /// Display one image pixel per logical UI point with scrollbars as needed.
    OriginalResolution,
}

/// The per-frame view the shared UI draws: the interaction state, the current
/// display texture (if a frame has been rendered), the render resolution, and an
/// optional last-render time for the status readout.
pub struct View<'a> {
    pub controller: &'a mut InteractionController,
    pub texture: Option<&'a TextureHandle>,
    pub render_size: (u32, u32),
    pub last_render_ms: Option<f32>,
    /// Set by [`show`] to `Some((x, y))` (render-target pixel coords) when the
    /// user **clicks** the image, requesting a pick (#141). The host app consumes
    /// it after `show`: runs the id pass, sets `controller.state.selected`, and
    /// re-renders. `None` when there was no click this frame.
    pub pick_request: &'a mut Option<(u32, u32)>,
}

/// Optional feature-specific additions around the standard viewer.
#[derive(Default)]
pub struct UiExtensions<'a> {
    pub top_controls: Option<&'a mut ExtraControls<'a>>,
    pub extra_controls: Option<&'a mut ExtraControls<'a>>,
    pub image_overlay: Option<&'a mut ImageOverlay<'a>>,
    pub camera_locked: bool,
    pub image_sizing: ImageSizing,
    pub move_reference_labels: Option<[&'static str; 3]>,
    pub hide_empty_image: bool,
    pub fitted_render_size: Option<&'a mut (u32, u32)>,
    pub central_bottom: Option<&'a mut CentralBottom<'a>>,
    pub central_bottom_height: Option<f32>,
}

/// Lays out the side controls + central image and maps input to interaction
/// events. Returns `true` if the scene changed and a re-render is needed.
pub fn show(ui: &mut egui::Ui, view: &mut View) -> bool {
    show_with_extensions(ui, view, &mut UiExtensions::default())
}

/// Lays out the shared viewer with feature-specific controls and image paint.
pub fn show_with_extensions(
    ui: &mut egui::Ui,
    view: &mut View,
    extensions: &mut UiExtensions,
) -> bool {
    let mut needs_render = false;
    egui::Panel::left("controls")
        .resizable(true)
        .default_size(264.0)
        .min_size(240.0)
        .max_size(420.0)
        .show(ui, |ui| {
            needs_render |= controls_panel(ui, view, extensions);
        });
    let bottom_height = extensions.central_bottom_height.unwrap_or(72.0);
    if let Some(bottom) = extensions.central_bottom.as_deref_mut() {
        egui::Panel::bottom("feature-central-bottom")
            .resizable(false)
            .default_size(bottom_height)
            .min_size(bottom_height)
            .max_size(bottom_height)
            .show(ui, |ui| {
                bottom(ui);
            });
    }
    egui::CentralPanel::default().show(ui, |ui| {
        needs_render |= image_panel(ui, view, extensions);
    });
    needs_render
}

/// The left control panel: primary-drag target, render mode, overlays, reset, and
/// a status readout. Grouped into collapsible sections inside a scroll area so the
/// (now taller) panel stays usable. Returns whether the scene changed.
fn controls_panel(ui: &mut egui::Ui, view: &mut View, extensions: &mut UiExtensions) -> bool {
    let mut needs_render = false;
    ui.horizontal(|ui| {
        ui.heading("trd-gui");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.weak(format!("proto {}", trd_core::PROTOCOL_VERSION));
        });
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        if let Some(top) = extensions.top_controls.as_deref_mut() {
            needs_render |= top(ui);
            ui.separator();
        }
        // ── Interaction ──────────────────────────────────────────────────
        needs_render |= section(ui, "Interaction", |ui| {
            let mut c = false;
            ui.label("Primary drag");
            let target = &mut view.controller.target;
            ui.horizontal_wrapped(|ui| {
                ui.add_enabled_ui(!extensions.camera_locked, |ui| {
                    c |= ui
                        .selectable_value(target, InteractionTarget::Camera, "Orbit camera")
                        .changed();
                });
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
                if view.controller.mode == TransformMode::Move {
                    ui.add_space(4.0);
                    ui.label("Translate direction");
                    if extensions.move_reference_labels.is_none() {
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::Free,
                            "Free",
                        );
                    }
                    let reference_labels = extensions
                        .move_reference_labels
                        .unwrap_or(["Parent X", "Parent Y", "Parent Z"]);
                    ui.label(if extensions.move_reference_labels.is_some() {
                        "Quad basis"
                    } else {
                        "Parent basis"
                    });
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::Reference1,
                            reference_labels[0],
                        );
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::Reference2,
                            reference_labels[1],
                        );
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::Reference3,
                            reference_labels[2],
                        );
                    });
                    ui.label("Object local basis");
                    ui.horizontal_wrapped(|ui| {
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::LocalX,
                            "X",
                        );
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::LocalY,
                            "Y",
                        );
                        ui.selectable_value(
                            &mut view.controller.move_direction,
                            MoveDirection::LocalZ,
                            "Z",
                        );
                    });
                }
                if view.controller.mode == TransformMode::Rotate {
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
        needs_render |= section(ui, "Transform", |ui| transform_panel(ui, view));

        // ── Render mode (per selected object) ────────────────────────────
        needs_render |= section(ui, "Render mode", |ui| {
            let mut c = false;
            match view.controller.state.selected_mode_mut() {
                Some(mode) => {
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
                }
                None => {
                    ui.weak("Select an object to set its render mode");
                }
            }
            c
        });

        // The Disney PBR material controls, only while the *selected* object is in
        // PBR mode.
        let selected_is_pbr = view
            .controller
            .state
            .selected
            .map(|i| view.controller.state.mode_of(i as usize) == RenderMode::Pbr)
            .unwrap_or(false);
        if selected_is_pbr {
            needs_render |= section(ui, "PBR material", |ui| pbr_panel(ui, view));
        }

        // ── Overlays ─────────────────────────────────────────────────────
        needs_render |= section(ui, "Overlays", |ui| {
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
            c |= ui
                .add_enabled(
                    state.environment_available,
                    egui::Checkbox::new(
                        &mut state.show_environment_background,
                        "Environment background",
                    ),
                )
                .changed();
            if state.environment_available && state.show_environment_background {
                ui.label("Background blur");
                c |= ui
                    .add(egui::Slider::new(
                        &mut state.environment_background_blur,
                        0.0..=1.0,
                    ))
                    .changed();
            }
            if state.environment_available {
                ui.label("Background / IBL rotation");
                if let Some(ibl) = state.image_based_lighting.first_mut() {
                    let mut degrees = ibl.rotation.to_degrees().rem_euclid(360.0);
                    if ui
                        .add(egui::Slider::new(&mut degrees, 0.0..=360.0).suffix("°"))
                        .changed()
                    {
                        ibl.rotation = degrees.to_radians();
                        c = true;
                    }
                }
            }
            c
        });

        // ── Selection ────────────────────────────────────────────────────
        needs_render |= section(ui, "Selection", |ui| {
            let mut c = false;
            match view.controller.state.selected {
                Some(idx) => {
                    ui.label(format!("Object #{idx} selected"));
                    if ui.button("Deselect").clicked() {
                        view.controller.state.selected = None;
                        c = true;
                    }
                }
                None => {
                    ui.weak("Click an object to select it");
                }
            }
            c
        });

        if let Some(extra) = extensions.extra_controls.as_deref_mut() {
            ui.separator();
            needs_render |= extra(ui);
        }

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
    // Transforms edit the *selected* object; with nothing selected there is
    // nothing to transform, so the widgets are hidden behind a hint (#141).
    let Some(obj) = view.controller.state.selected_object_mut() else {
        ui.weak("Select an object to transform it");
        return false;
    };
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
    // The PBR material is per-object (#141): edit the *selected* object's
    // material; with nothing selected there is no material to edit.
    let Some((material, ibl, tone_mapping, debug_view)) = view.controller.state.selected_pbr_mut()
    else {
        ui.weak("Select an object to edit its material");
        return false;
    };
    ui.label("PBR view");
    ui.horizontal_wrapped(|ui| {
        changed |= ui
            .selectable_value(debug_view, PbrDebugView::Shaded, "Shaded")
            .changed();
        changed |= ui
            .selectable_value(debug_view, PbrDebugView::Roughness, "Roughness")
            .changed();
        changed |= ui
            .selectable_value(debug_view, PbrDebugView::Metallic, "Metallic")
            .changed();
        changed |= ui
            .selectable_value(debug_view, PbrDebugView::Normal, "Normal")
            .changed();
    });
    // Label each slider on its own line so the text never clips in a narrow
    // panel; the slider then spans the full panel width beneath it.
    let mapped_metallic_roughness = material.auxiliary.textures.metallic_roughness;
    ui.label(if mapped_metallic_roughness {
        "Metallic factor"
    } else {
        "Metallic"
    });
    changed |= ui
        .add(egui::Slider::new(&mut material.metallic, 0.0..=1.0))
        .changed();
    ui.label(if mapped_metallic_roughness {
        "Roughness factor"
    } else {
        "Roughness"
    });
    changed |= ui
        .add(egui::Slider::new(&mut material.roughness, 0.0..=1.0))
        .changed();
    if mapped_metallic_roughness {
        ui.weak("Factors multiply the imported GLB metallic-roughness map");
    }
    ui.label("Clearcoat");
    changed |= ui
        .add(egui::Slider::new(&mut material.clearcoat, 0.0..=1.0))
        .changed();
    ui.label("Env intensity");
    changed |= ui
        .add(egui::Slider::new(&mut ibl.intensity, 0.0..=4.0))
        .changed();
    ui.label("Exposure");
    changed |= ui
        .add(egui::Slider::new(&mut tone_mapping.exposure, 0.0..=4.0))
        .changed();
    ui.label("Tonemap");
    ui.horizontal_wrapped(|ui| {
        changed |= ui
            .selectable_value(&mut tone_mapping.operator, Tonemap::Reinhard, "Reinhard")
            .changed();
        changed |= ui
            .selectable_value(&mut tone_mapping.operator, Tonemap::Aces, "ACES")
            .changed();
    });
    changed
}
/// pointer/scroll input over it into interaction events. Returns whether the
/// scene changed.
fn image_panel(ui: &mut egui::Ui, view: &mut View, extensions: &mut UiExtensions) -> bool {
    let Some(texture) = view.texture else {
        if !extensions.hide_empty_image {
            ui.centered_and_justified(|ui| {
                ui.label("Rendering…");
            });
        }
        return false;
    };

    let (img_w, img_h) = view.render_size;
    let img_aspect = img_w as f32 / img_h.max(1) as f32;
    let avail = ui.available_size();
    let fit = || {
        if avail.x / avail.y.max(1.0) > img_aspect {
            Vec2::new(avail.y * img_aspect, avail.y)
        } else {
            Vec2::new(avail.x, avail.x / img_aspect)
        }
    };
    let add_image = |ui: &mut egui::Ui, size| {
        ui.add(
            egui::Image::new(egui::load::SizedTexture::from_handle(texture))
                .fit_to_exact_size(size)
                .sense(Sense::click_and_drag()),
        )
    };
    let response = match extensions.image_sizing {
        ImageSizing::FitCanvas => {
            let display_size = fit();
            if let Some(size) = extensions.fitted_render_size.as_deref_mut() {
                let pixels_per_point = ui.ctx().pixels_per_point();
                *size = (
                    (display_size.x * pixels_per_point).round().max(1.0) as u32,
                    (display_size.y * pixels_per_point).round().max(1.0) as u32,
                );
            }
            ui.centered_and_justified(|ui| add_image(ui, display_size))
                .inner
        }
        ImageSizing::OriginalResolution => {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    let image_size = Vec2::new(img_w as f32, img_h as f32);
                    let canvas_size =
                        Vec2::new(image_size.x.max(avail.x), image_size.y.max(avail.y));
                    ui.allocate_ui_with_layout(
                        canvas_size,
                        egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                        |ui| add_image(ui, image_size),
                    )
                    .inner
                })
                .inner
        }
    };

    if let Some(overlay) = extensions.image_overlay.as_mut() {
        overlay(ui, response.rect);
    }

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

    // A primary click (press + release without dragging) requests a pick: map the
    // pointer position within the letterboxed image to render-target pixel coords
    // for the host app to resolve into a selection (#141).
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let (rw, rh) = view.render_size;
            let u = ((pos.x - response.rect.min.x) / size.x).clamp(0.0, 1.0);
            let v = ((pos.y - response.rect.min.y) / size.y).clamp(0.0, 1.0);
            let px = ((u * rw as f32) as u32).min(rw.saturating_sub(1));
            let py = ((v * rh as f32) as u32).min(rh.saturating_sub(1));
            *view.pick_request = Some((px, py));
            changed = true;
        }
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
                Some(InteractionEvent::Scale { delta })
            } else if extensions.camera_locked {
                None
            } else {
                Some(InteractionEvent::Zoom { delta })
            };
            if let Some(event) = event {
                changed |= view.controller.apply(event);
            }
        }
    }

    if changed {
        ui.ctx().request_repaint();
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::section;

    /// Runs `body` inside one headless egui frame (CPU layout only — no window /
    /// GPU) and returns its result, so panel helpers can be unit-tested.
    fn in_frame<R>(body: impl FnOnce(&mut egui::Ui) -> R) -> R {
        let mut out = None;
        let mut body = Some(body);
        // A default-open `CollapsingHeader` runs its body on the first frame, so a
        // single test-ui pass is enough to exercise `section`.
        egui::__run_test_ui(|ui| {
            if let Some(body) = body.take() {
                out = Some(body(ui));
            }
        });
        out.expect("test ui body ran")
    }

    #[test]
    fn section_folds_out_its_body_changed_flag() {
        // Regression: the panel-polish refactor dropped `section`'s returned
        // "changed" flag, so control edits (e.g. PBR sliders) didn't re-render
        // until the next image drag. `section` must propagate its body's bool so
        // the caller's `needs_render` accounting still fires.
        assert!(
            in_frame(|ui| section(ui, "changed", |_| true)),
            "a section whose body reports a change must return true"
        );
        assert!(
            !in_frame(|ui| section(ui, "unchanged", |_| false)),
            "a section whose body reports no change must return false"
        );
    }
}
