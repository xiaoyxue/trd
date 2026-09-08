//! The editor's egui surface (#163/#167): left-pane editing controls,
//! source-model interaction, and player footer.

use super::details_ui::details_ui;
use super::{point_in_quad, VideoEditingApp, VideoSourceKind, COMMAND_PAUSE, COMMAND_PLAY};

impl eframe::App for VideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.shared.context.replace(Some(ui.ctx().clone()));
        if let Some(video) = self.shared.take_pending_video_info() {
            // A timeline probed from the container after start-up (#264).
            self.set_video_info(video);
        }
        if let Some(document) = self.shared.take_incoming_document() {
            self.set_document(document);
        }
        if let Some(scene) = self.shared.take_incoming_scene() {
            self.set_arrow_scene(scene);
        }
        self.consume_video_frame();
        self.consume_rendered_frame();
        self.consume_asset_defaults();
        self.consume_pick_result();
        self.process_scene_operations();
        self.sync_native_texture(_frame);
        if !self.shared.video_loaded.get() {
            self.displayed_frame_ready = false;
            self.last_rendered_frame_index = None;
            self.displayed_diagnostics = None;
            self.pending_seek = None;
            self.last_pick_result = None;
        }
        if self.shared.latest_video_frame.borrow().is_some() {
            self.ensure_texture(ui.ctx());
        }
        let playing = self.shared.video_playing.get();
        self.was_playing = playing;
        self.schedule_pick();
        self.schedule_overlay();

        self.video_source_dialog(ui.ctx());

        let overlay_frame_index = self.displayed_frame_index;
        let source_mode = self
            .arrow_scene
            .as_ref()
            .is_some_and(|scene| scene.source.is_some());
        if source_mode {
            if let Err(error) = self.sync_source_controller() {
                self.shared.set_error(super::ErrorScope::Document, error);
            }
        }
        let quad = if source_mode {
            self.displayed_diagnostics
                .as_ref()
                .and_then(|facts| facts.source_frame.as_ref())
                .and_then(|frame| frame.objects.get(self.source_selected_instance))
                .and_then(|object| object.quad)
        } else {
            self.frame_row(overlay_frame_index)
                .and_then(|frame| frame.placement_quad)
        };
        let quad_frame = if source_mode {
            self.displayed_facts().quad
        } else {
            self.quad_frame_at(overlay_frame_index)
        };
        let mut needs_render = false;
        let mut pick = None;
        let mut hover = None;

        egui::Panel::left("controls")
            .resizable(true)
            .default_size(264.0)
            .min_size(240.0)
            .max_size(420.0)
            .show(ui, |ui| {
                crate::ui::header(ui);
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.source_controls(ui);
                    ui.separator();
                    needs_render |= self.source_model_controls(ui);
                    ui.separator();
                    self.export_controls(ui);
                    self.details_controls(ui);
                    needs_render |= crate::ui::reset_button(ui, &mut self.controller);
                    if ui
                        .button("Reset all")
                        .on_hover_text(
                            "Clear the quad selection, the placed object, its transform and \
                             material, and restore the overlay toggles. Keeps the video and \
                             document.",
                        )
                        .clicked()
                    {
                        self.reset_all();
                        needs_render = true;
                    }
                    ui.separator();
                    crate::ui::status(ui, self.display_size, None);
                });
            });

        egui::Panel::bottom("video-editing-player")
            .resizable(false)
            .default_size(80.0)
            .min_size(80.0)
            .max_size(80.0)
            .show(ui, |ui| self.player_controls(ui));

        egui::CentralPanel::default().show(ui, |ui| {
            let texture = if !self.shared.video_loaded.get() {
                None
            } else if let Some(id) = self.native_texture {
                Some(crate::ui::DisplayTexture::Native {
                    id,
                    size: self.display_size,
                })
            } else {
                self.display_texture
                    .as_ref()
                    .map(crate::ui::DisplayTexture::Uploaded)
            };
            let outcome = crate::ui::image_panel(
                ui,
                crate::ui::Image {
                    controller: &mut self.controller,
                    texture,
                    render_size: self.display_size,
                    sizing: crate::ui::ImageSizing::FitCanvas,
                    camera_locked: true,
                    hide_when_empty: true,
                },
            );
            needs_render |= outcome.needs_render;
            if let Some(fitted) = outcome.fitted_size {
                self.resize_render_target(ui.ctx(), fitted);
            }
            pick = outcome.pick;
            hover = outcome.hover;
            if let Some(rect) = outcome.image_rect {
                self.paint_axis_labels(ui, rect, overlay_frame_index, quad_frame);
            }
        });

        self.settle_frame(ui.ctx(), needs_render, pick, hover, quad);
    }
}

impl VideoEditingApp {
    fn source_model_controls(&mut self, ui: &mut egui::Ui) -> bool {
        ui.heading("Placement / source models");
        let Some(scene) = self.arrow_scene.as_ref() else {
            ui.weak("Load params Arrow with optional GLB mesh resources. Video playback is independent.");
            return false;
        };
        let Some(source) = scene.source.clone() else {
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "This scene is not a current params/GLB document.",
            );
            return false;
        };
        let mut overlay_changed = ui
            .checkbox(&mut self.show_placement_quads, "Show quad")
            .changed();
        overlay_changed |= ui
            .add_enabled(
                self.selected_quad,
                egui::Checkbox::new(&mut self.show_gizmos, "Show Coordinate"),
            )
            .changed();
        overlay_changed |= ui
            .add_enabled(
                self.selected_quad,
                egui::Checkbox::new(&mut self.show_plane_grid, "Show Plane"),
            )
            .changed();
        let reference_only = source.borrow().meshes().is_empty();
        if reference_only {
            overlay_changed |= ui
                .add_enabled(
                    self.selected_quad,
                    egui::Checkbox::new(&mut self.show_reference_cube, "Show reference cube"),
                )
                .changed();
        }
        let Some(row) = scene.source_row(self.displayed_frame_index) else {
            ui.weak("No params row for this video frame.");
            return overlay_changed;
        };
        if reference_only {
            ui.weak(if self.selected_quad {
                "Selected quad: the cube's bottom center is at the local origin."
            } else {
                "Click a quad to show its local coordinates, plane grid and cube."
            });
            return overlay_changed;
        }
        let frame = match source.borrow().frame(row) {
            Ok(frame) => frame,
            Err(error) => {
                self.shared
                    .set_error(super::ErrorScope::Document, error.to_string());
                return false;
            }
        };
        if frame.objects.is_empty() {
            ui.weak("This params row has no objects.");
            return false;
        }
        self.source_selected_instance = self.source_selected_instance.min(frame.objects.len() - 1);
        let mut changed = false;
        ui.add_enabled_ui(!self.shared.video_playing.get(), |ui| {
            let label = |index: usize| {
                frame.objects[index]
                    .track_id
                    .clone()
                    .unwrap_or_else(|| format!("Object {}", index + 1))
            };
            let previous = self.source_selected_instance;
            egui::ComboBox::from_id_salt("source-instance")
                .selected_text(label(self.source_selected_instance))
                .show_ui(ui, |ui| {
                    for index in 0..frame.objects.len() {
                        ui.selectable_value(
                            &mut self.source_selected_instance,
                            index,
                            label(index),
                        );
                    }
                });
            if previous != self.source_selected_instance {
                self.controller.state.selected = Some(self.source_selected_instance as u32);
                self.selected_quad = true;
                self.show_gizmos = true;
                self.show_plane_grid = true;
                changed = true;
            }
            ui.weak(format!(
                "Params row {row}; edits affect this object across all its tracked frames."
            ));
            ui.weak("Pick the mesh to edit it. Controls adjust its existing local model.");
            changed |= crate::ui::interaction_section(
                ui,
                &mut self.controller,
                crate::ui::Controls {
                    camera_locked: true,
                    move_reference_labels: Some(super::QUAD_MOVE_LABELS),
                },
            );
            changed |= crate::ui::transform_section(ui, &mut self.controller);
            ui.collapsing("Model matrix (advanced)", |ui| {
                ui.monospace(super::diagnostics::format_matrix(
                    frame.objects[self.source_selected_instance]
                        .model
                        .to_cols_array(),
                ));
            });
        });
        changed || overlay_changed
    }

    /// Scene revision settles before the pick captures it (#205).
    pub(super) fn settle_frame(
        &mut self,
        ctx: &egui::Context,
        needs_render: bool,
        pick: Option<(u32, u32)>,
        hover: Option<(u32, u32)>,
        quad: Option<[[f32; 2]; 4]>,
    ) {
        if needs_render {
            if let Err(error) = self.apply_source_transforms() {
                self.shared.set_error(super::ErrorScope::Document, error);
                return;
            }
            self.shared.request_overlay();
            ctx.request_repaint();
        }
        self.update_quad_hover(hover, quad);
        if let Some(point) = pick {
            self.handle_pick(point, quad);
        }
    }

    /// Updates hover state; only re-renders on a change.
    fn update_quad_hover(&mut self, hover: Option<(u32, u32)>, quad: Option<[[f32; 2]; 4]>) {
        if self
            .arrow_scene
            .as_ref()
            .is_some_and(|scene| scene.source.is_some())
        {
            let hovered = hover.and_then(|point| {
                self.displayed_diagnostics
                    .as_ref()
                    .and_then(|facts| facts.source_frame.as_ref())
                    .and_then(|frame| {
                        frame
                            .objects
                            .iter()
                            .enumerate()
                            .rev()
                            .find(|(_, object)| self.point_hits_quad(point, object.quad))
                            .map(|(index, _)| index)
                    })
            });
            if hovered != self.source_hovered_instance {
                self.source_hovered_instance = hovered;
                self.hovered_quad = hovered.is_some();
                self.shared.request_overlay();
                self.shared.request_repaint();
            }
            return;
        }
        let hovered = hover.is_some_and(|point| self.point_hits_quad(point, quad));
        if hovered != self.hovered_quad {
            self.hovered_quad = hovered;
            self.shared.request_overlay();
            self.shared.request_repaint();
        }
    }

    /// Whether a render-target pixel lands inside the tracked quad, mapping it
    /// back to source-video coordinates first.
    fn point_hits_quad(&self, (x, y): (u32, u32), quad: Option<[[f32; 2]; 4]>) -> bool {
        quad.is_some_and(|points| {
            let source_size = self
                .displayed_diagnostics
                .as_ref()
                .and_then(|facts| facts.source_frame.as_ref())
                .and_then(|frame| frame.source_size)
                .unwrap_or(trd_core::Viewport {
                    width: self.video.width,
                    height: self.video.height,
                });
            let source = [
                x as f32 * source_size.width as f32 / self.display_size.0 as f32,
                y as f32 * source_size.height as f32 / self.display_size.1 as f32,
            ];
            point_in_quad(source, points)
        })
    }

    /// Source heading, the Open button, and the loaded-source readout.
    fn source_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Video");
        ui.horizontal(|ui| {
            if ui.button("Open Video").clicked() {
                self.source_dialog = Some(super::SourceDialog::Video);
                ui.ctx().request_repaint();
            }
            if ui.button("Load Arrow").clicked() {
                self.source_dialog = Some(super::SourceDialog::Arrow);
                ui.ctx().request_repaint();
            }
        });
        ui.weak("Display: fit right pane (16:9)");
        ui.collapsing("Source", |ui| {
            let video = &self.video;
            ui.label(format!("Source: {}", video.source_name));
            ui.label(match self.shared.pending_document() {
                Some(source) => match source.kind {
                    VideoSourceKind::LocalFile => format!("Arrow input: {} (local)", source.name),
                    VideoSourceKind::HttpUrl => format!("Arrow input: {}", source.name),
                },
                None if self.arrow_scene.is_some() => {
                    "Arrow input: source params loaded".to_owned()
                }
                None if self.document.is_some() => "Arrow input: annotation loaded".to_owned(),
                None => "Arrow input: none — the video plays as-is".to_owned(),
            });
            match (self.document.as_ref(), self.arrow_scene.as_ref()) {
                (Some(document), _) => {
                    let summary = super::document_summary(document, &self.video);
                    ui.label(summary.describes);
                    ui.label(summary.annotated);
                    if let Some(mismatch) = summary.mismatch {
                        ui.colored_label(egui::Color32::from_rgb(240, 180, 80), mismatch);
                    }
                }
                (None, Some(scene)) => {
                    ui.label(format!(
                        "Protocol {} scene · {} params rows · {:.6} fps",
                        trd_core::PROTOCOL_VERSION,
                        scene.frames.len(),
                        scene.frame_rate
                    ));
                    ui.weak(if scene.source.is_some() {
                        "Source editing: model changes preserve the other Arrow columns."
                    } else {
                        "Replay mode: the exported models are rendered over this video."
                    });
                }
                (None, None) => {
                    ui.weak("No document loaded: every frame is plain video");
                }
            }
            ui.label(format!(
                "{}x{} · {}/{} fps · {} frames",
                video.width, video.height, video.fps_num, video.fps_den, video.frame_count
            ));
            ui.label(
                match (
                    self.shared.video_loaded.get(),
                    self.shared.video_playing.get(),
                ) {
                    (false, _) => "No video loaded",
                    (true, true) => "Playing video",
                    (true, false) => "Video paused",
                },
            );
            if let Some(error) = self.shared.error_text() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
        });
    }

    fn export_controls(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.strong("Scene export");
            let disabled = self.arrow_export_disabled_reason();
            if ui
                .add_enabled(disabled.is_none(), egui::Button::new("Export Arrow..."))
                .on_disabled_hover_text(disabled.as_deref().unwrap_or_default())
                .clicked()
            {
                self.request_arrow_export();
            }
            match self.arrow_export_status() {
                Some(Ok(message)) => {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, message);
                }
                Some(Err(error)) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
                None => {
                    ui.weak("Params and original GLB resources; video remains external.");
                }
            }
            ui.weak("Edits persist in all corresponding sparse-frame model matrices.");
            ui.weak("Replay uses uffizi-large.hdr as the default IBL probe.");
        });
    }

    /// The Details inspector.
    fn details_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Details", |ui| {
            let facts = self.displayed_facts();
            details_ui(ui, &self.video, &facts);
        });
    }

    /// Play/pause, the time readout, and the scrub bar.
    fn player_controls(&mut self, ui: &mut egui::Ui) {
        let video_loaded = self.shared.video_loaded.get();
        let video_playing = self.shared.video_playing.get();
        ui.add_space(4.0);
        let (row_rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::hover());
        let button_rect = egui::Rect::from_center_size(row_rect.center(), egui::vec2(64.0, 28.0));
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(button_rect)
                .layout(egui::Layout::top_down(egui::Align::Center)),
            |ui| {
                if video_playing {
                    if ui.button("Pause").clicked() {
                        self.shared.command.set(COMMAND_PAUSE);
                    }
                } else if ui
                    .add_enabled(video_loaded, egui::Button::new("Play"))
                    .clicked()
                {
                    self.shared.command.set(COMMAND_PLAY);
                }
            },
        );
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(row_rect)
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
            |ui| {
                ui.monospace(player_status_label(
                    video_loaded,
                    self.current_frame_index,
                    &self.video,
                ));
            },
        );
        ui.add_space(6.0);
        let last = self.video.frame_count.saturating_sub(1);
        let mut frame = self.current_frame_index;
        if video_progress_bar(ui, &mut frame, last, video_loaded) {
            self.seek_to(frame);
        }
    }

    fn seek_to(&mut self, frame_index: u32) {
        if frame_index == self.current_frame_index {
            return;
        }
        self.begin_seek(frame_index);
    }

    /// Labels basis arms `e1`/`e2`/`e3` at their tips using projected egui text.
    fn paint_axis_labels(
        &self,
        ui: &egui::Ui,
        rect: egui::Rect,
        frame_index: u32,
        quad_frame: Option<trd_placement::QuadFrame>,
    ) {
        if !self.show_gizmos {
            return;
        }
        let Some(frame) = quad_frame else {
            return;
        };
        let Some(k) = self
            .displayed_diagnostics
            .as_ref()
            .and_then(|facts| facts.source_frame.as_ref())
            .and_then(|frame| frame.params.k)
            .map(super::protocol_k_from_row_major)
            .or_else(|| self.frame_row(frame_index).and_then(|row| row.k))
        else {
            return;
        };
        let intrinsics = trd_placement::CameraIntrinsics { row_major: k };
        let tips = [
            ("e1", frame.half_edge1, egui::Color32::from_rgb(255, 70, 70)),
            ("e2", frame.half_edge2, egui::Color32::from_rgb(20, 200, 45)),
            (
                "e3",
                [
                    frame.e3[0] * frame.axis_length,
                    frame.e3[1] * frame.axis_length,
                    frame.e3[2] * frame.axis_length,
                ],
                egui::Color32::from_rgb(45, 105, 245),
            ),
        ];
        let painter = ui.painter_at(rect);
        for (label, arm, color) in tips {
            let tip = [
                frame.origin_camera[0] + arm[0],
                frame.origin_camera[1] + arm[1],
                frame.origin_camera[2] + arm[2],
            ];
            let Ok([x, y]) = trd_placement::project_camera(intrinsics, tip) else {
                continue;
            };
            let position = egui::pos2(
                rect.min.x + x / self.video.width as f32 * rect.width(),
                rect.min.y + y / self.video.height as f32 * rect.height(),
            );
            if !rect.contains(position) {
                continue;
            }
            let font = egui::FontId::proportional(20.0);
            painter.text(
                position + egui::vec2(1.5, 1.5),
                egui::Align2::LEFT_TOP,
                label,
                font.clone(),
                egui::Color32::from_black_alpha(220),
            );
            painter.text(position, egui::Align2::LEFT_TOP, label, font, color);
        }
    }

    /// Re-imposes the source aspect so the render target stays the video's shape (#282).
    fn resize_render_target(&mut self, ctx: &egui::Context, fitted: (u32, u32)) {
        let fitted = fit_to_source_aspect(fitted, (self.video.width, self.video.height));
        if self.render_sizing != crate::ui::ImageSizing::FitCanvas
            || fitted == self.fitted_render_size
        {
            return;
        }
        self.fitted_render_size = fitted;
        if self.selected_asset.is_some() {
            self.shared.request_overlay();
            ctx.request_repaint();
        }
    }

    /// Resolves a click on the image into a quad selection or a GPU pick request.
    ///
    /// Resolves a click into a quad selection or GPU pick. Selection bumps before
    /// pick so the pick captures the updated revision (#205).
    pub(super) fn handle_pick(&mut self, (x, y): (u32, u32), quad: Option<[[f32; 2]; 4]>) {
        if self
            .arrow_scene
            .as_ref()
            .and_then(|scene| scene.source.as_ref())
            .is_some_and(|source| !source.borrow().meshes().is_empty())
        {
            if !self.shared.video_playing.get() {
                self.shared.request_pick((x, y));
            }
            return;
        }
        if self
            .arrow_scene
            .as_ref()
            .is_some_and(|scene| scene.source.is_some())
        {
            if self.shared.video_playing.get() {
                return;
            }
            let selected = self
                .displayed_diagnostics
                .as_ref()
                .and_then(|facts| facts.source_frame.as_ref())
                .and_then(|frame| {
                    frame
                        .objects
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, object)| self.point_hits_quad((x, y), object.quad))
                        .map(|(index, _)| index)
                });
            if self.selected_quad != selected.is_some()
                || selected.is_some_and(|index| index != self.source_selected_instance)
            {
                self.selected_quad = selected.is_some();
                if let Some(index) = selected {
                    self.source_selected_instance = index;
                }
                self.show_gizmos = self.selected_quad;
                self.show_plane_grid = self.selected_quad;
                self.show_reference_cube = self.selected_quad;
                self.shared.request_overlay();
            }
            return;
        }
        if self.selected_asset.is_some() {
            self.shared.request_pick((x, y));
            return;
        }
        if self.shared.video_playing.get() {
            return;
        }
        let clicked_quad = self.point_hits_quad((x, y), quad);
        if clicked_quad != self.selected_quad {
            self.selected_quad = clicked_quad;
            self.show_gizmos = clicked_quad;
            self.shared.request_overlay();
        }
        if self.selected_quad {
            self.controller.target = crate::interaction::InteractionTarget::Object;
        }
    }
}

fn video_progress_bar(ui: &mut egui::Ui, frame: &mut u32, last_frame: u32, enabled: bool) -> bool {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 18.0),
        if enabled {
            egui::Sense::click_and_drag()
        } else {
            egui::Sense::hover()
        },
    );
    let mut changed = false;
    if enabled && (response.clicked() || response.dragged()) {
        if let Some(pointer) = response.interact_pointer_pos() {
            let fraction = ((pointer.x - rect.left()) / rect.width().max(1.0)).clamp(0.0, 1.0);
            let next = (fraction * last_frame as f32).round() as u32;
            changed = next != *frame;
            *frame = next;
        }
    }

    let visuals = ui.visuals();
    let track = egui::Rect::from_center_size(rect.center(), egui::vec2(rect.width(), 6.0));
    let fraction = if last_frame == 0 {
        0.0
    } else {
        *frame as f32 / last_frame as f32
    };
    let knob_x = egui::lerp(track.left()..=track.right(), fraction);
    let played = egui::Rect::from_min_max(track.min, egui::pos2(knob_x, track.bottom()));
    let background = if enabled {
        visuals.widgets.inactive.bg_fill
    } else {
        visuals.widgets.noninteractive.bg_fill
    };
    let accent = visuals.selection.bg_fill;
    ui.painter().rect_filled(track, 3.0, background);
    ui.painter().rect_filled(played, 3.0, accent);
    ui.painter().circle_filled(
        egui::pos2(knob_x, track.center().y),
        if response.hovered() { 6.0 } else { 5.0 },
        if enabled {
            accent
        } else {
            visuals.weak_text_color()
        },
    );
    changed
}

fn media_time_label(frame: u32, fps_num: u32, fps_den: u32) -> String {
    let seconds = u64::from(frame) * u64::from(fps_den) / u64::from(fps_num.max(1));
    format!("{:02}:{:02}", seconds / 60, seconds % 60)
}

fn player_status_label(loaded: bool, frame: u32, video: &trd_core::VideoInfo) -> String {
    if !loaded {
        return "00:00 / 00:00  ·  frame 0/0".to_owned();
    }
    let current = media_time_label(frame, video.fps_num, video.fps_den);
    let total = media_time_label(video.frame_count, video.fps_num, video.fps_den);
    format!(
        "{current} / {total}  ·  frame {}/{}",
        frame.saturating_add(1),
        video.frame_count
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unloaded_player_status_is_zeroed() {
        let document = super::super::tests::document();
        assert_eq!(
            player_status_label(false, 42, &document.video),
            "00:00 / 00:00  ·  frame 0/0"
        );
    }
}

/// Fits `source`'s aspect inside `pane`; keeps the render target the video's shape (#282).
fn fit_to_source_aspect(pane: (u32, u32), source: (u32, u32)) -> (u32, u32) {
    let (pane_w, pane_h) = (pane.0.max(1), pane.1.max(1));
    let (source_w, source_h) = source;
    if source_w == 0 || source_h == 0 {
        return (pane_w, pane_h);
    }
    // Round rather than truncate: rounding makes the size a fixed point; truncation
    // walks the target smaller each frame (1493 → 1491 → 1489…).
    let scale = |value: u32, num: u32, den: u32| {
        let den = u64::from(den);
        u32::try_from((u64::from(value) * u64::from(num) + den / 2) / den).unwrap_or(u32::MAX)
    };
    let by_width = (pane_w, scale(pane_w, source_h, source_w).max(1));
    let by_height = (scale(pane_h, source_w, source_h).max(1), pane_h);
    let fits = |c: (u32, u32)| c.0 <= pane_w && c.1 <= pane_h;
    let area = |c: (u32, u32)| u64::from(c.0) * u64::from(c.1);
    let fitted = match (fits(by_width), fits(by_height)) {
        (true, true) if area(by_height) > area(by_width) => by_height,
        (false, true) => by_height,
        _ => by_width,
    };
    (fitted.0.min(source_w).max(1), fitted.1.min(source_h).max(1))
}

#[cfg(test)]
mod fit_tests {
    use super::fit_to_source_aspect;

    #[test]
    fn full_pane_is_letterboxed_to_the_source_aspect() {
        assert_eq!(
            fit_to_source_aspect((1493, 1080), (1920, 1080)),
            (1493, 840)
        );
    }

    #[test]
    fn repeated_application_does_not_drift() {
        let source = (1920, 1080);
        let mut size = fit_to_source_aspect((1493, 1080), source);
        for _ in 0..32 {
            let next = fit_to_source_aspect(size, source);
            assert_eq!(next, size, "drifted from {size:?} to {next:?}");
            size = next;
        }
    }

    #[test]
    fn a_taller_pane_is_width_bound_and_a_wider_pane_height_bound() {
        assert_eq!(fit_to_source_aspect((800, 1000), (1920, 1080)), (800, 450));
        assert_eq!(
            fit_to_source_aspect((4000, 1000), (1920, 1080)),
            (1778, 1000)
        );
    }

    #[test]
    fn never_upscales_past_the_source() {
        assert_eq!(
            fit_to_source_aspect((4000, 4000), (1920, 1080)),
            (1920, 1080)
        );
    }

    #[test]
    fn an_unknown_source_leaves_the_pane_alone() {
        assert_eq!(fit_to_source_aspect((0, 0), (0, 0)), (1, 1));
        assert_eq!(fit_to_source_aspect((100, 100), (0, 0)), (100, 100));
    }
}
