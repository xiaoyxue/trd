//! The editor's own egui surface (#163/#167): the left-pane editing controls,
//! the quad/catalog interaction wiring, and the player footer.
//!
//! This is the *editing* UI. The read-only Details inspector lives in
//! [`super::details_ui`], and the shared viewer controls (Interaction,
//! Transform, Render mode, PBR material, Overlays, Selection) stay in
//! [`crate::ui`]; this module composes them for the video editor and owns the
//! playback widgets.

use super::details_ui::details_ui;
use super::{point_in_quad, CatalogAsset, VideoEditingApp, COMMAND_PAUSE, COMMAND_PLAY};

impl eframe::App for VideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.shared.context.replace(Some(ui.ctx().clone()));
        self.consume_video_frame();
        self.consume_rendered_frame();
        self.consume_asset_defaults();
        self.consume_pick_result();
        if !self.shared.video_loaded.get() {
            self.displayed_frame_ready = false;
            self.last_rendered_frame_index = None;
            self.displayed_diagnostics = None;
            self.pending_seek_target = None;
            self.last_pick_result = None;
        }
        if self.shared.latest_video_frame.borrow().is_some() {
            self.ensure_texture(ui.ctx());
        }
        let playing = self.shared.video_playing.get();
        if playing && !self.was_playing {
            self.show_quad_gizmo = false;
            self.shared.request_overlay();
        }
        self.was_playing = playing;
        self.schedule_pick();
        self.schedule_overlay();

        self.video_source_dialog(ui.ctx());

        let overlay_frame_index = self.displayed_frame_index;
        let timeline_frame = &self.document.frames[overlay_frame_index as usize];
        let quad = timeline_frame.placement_quad;
        let quad_frame = self.quad_frame_at(overlay_frame_index);
        let selected_quad = self.selected_quad;
        let show_quad_gizmo = self.show_quad_gizmo;
        let selected_asset = self.selected_asset;
        let video_loaded = self.shared.video_loaded.get();
        let video_playing = self.shared.video_playing.get();
        let video = &self.document.video;
        let error = self.shared.error.borrow().clone();
        // Immediate mode already skips a collapsed body, so the facts are only
        // derived while the panel is open; no "is it open?" flag is needed.
        let details_facts = self.details_open.then(|| self.displayed_facts());
        let mut details_open = self.details_open;
        let mut requested_asset = None;
        let mut open_video_requested = false;
        let mut top_controls = |ui: &mut egui::Ui| {
            ui.heading("Video");
            if ui.button("Open video...").clicked() {
                open_video_requested = true;
            }
            ui.weak("Display: fit right pane (16:9)");
            ui.collapsing("Source", |ui| {
                ui.label(format!("Source: {}", video.source_name));
                ui.label(format!(
                    "{}x{} · {}/{} fps · {} frames",
                    video.width, video.height, video.fps_num, video.fps_den, video.frame_count
                ));
                ui.label(if video_loaded {
                    if video_playing {
                        "Playing video"
                    } else {
                        "Video paused"
                    }
                } else {
                    "No video loaded"
                });
                if let Some(error) = error.as_deref() {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            });
            false
        };
        let mut extra_controls = |ui: &mut egui::Ui| {
            let mut changed = false;
            ui.collapsing("Placement quad (standalone)", |ui| {
                ui.label(format!("Frame {}", timeline_frame.video_frame_index));
                ui.label(if timeline_frame.tracked {
                    if video_playing {
                        "Placement quad hidden during playback"
                    } else if selected_quad {
                        if show_quad_gizmo {
                            "Placement quad selected; gizmo visible"
                        } else {
                            "Placement quad selected; click it to show gizmo"
                        }
                    } else {
                        "Click the green quad to select it"
                    }
                } else {
                    "Background-only row: quad and object hidden"
                });
                if let Some(local) = quad_frame {
                    ui.label(format!("Local axis length: {:.4}", local.axis_length));
                    ui.weak("RGB axes: e1 / e2 / e3");
                    ui.weak("Quad overlay follows the displayed tracking row.");
                    ui.weak("Object edit state persists; quad basis updates per frame.");
                    ui.weak("Local X/Y/Z rotate with the placed object.");
                    ui.weak("Initial can placement matches the Olympic upper-can preset.");
                }
            });
            ui.add_enabled_ui(selected_quad, |ui| {
                ui.collapsing("Object catalog", |ui| {
                    for asset in CatalogAsset::ALL {
                        if ui
                            .selectable_label(selected_asset == Some(asset), asset.label())
                            .clicked()
                        {
                            requested_asset = Some(asset);
                            changed = true;
                        }
                    }
                });
            });
            let response = ui.collapsing("Details", |ui| match details_facts.as_ref() {
                Some(facts) => details_ui(ui, video, facts),
                None => {
                    ui.weak("Loading details...");
                    ui.ctx().request_repaint();
                }
            });
            details_open = response.fully_open();
            changed
        };
        let mut requested_frame = self.current_frame_index;
        let mut playback_command = None;
        let mut central_bottom = |ui: &mut egui::Ui| {
            ui.add_space(4.0);
            let (row_rect, _) = ui
                .allocate_exact_size(egui::vec2(ui.available_width(), 28.0), egui::Sense::hover());
            let button_rect =
                egui::Rect::from_center_size(row_rect.center(), egui::vec2(64.0, 28.0));
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(button_rect)
                    .layout(egui::Layout::top_down(egui::Align::Center)),
                |ui| {
                    if video_playing {
                        if ui.button("Pause").clicked() {
                            playback_command = Some(COMMAND_PAUSE);
                        }
                    } else if ui
                        .add_enabled(video_loaded, egui::Button::new("Play"))
                        .clicked()
                    {
                        playback_command = Some(COMMAND_PLAY);
                    }
                },
            );
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(row_rect)
                    .layout(egui::Layout::right_to_left(egui::Align::Center)),
                |ui| {
                    ui.monospace(player_status_label(video_loaded, requested_frame, video));
                },
            );
            ui.add_space(6.0);
            let last = video.frame_count.saturating_sub(1);
            video_progress_bar(ui, &mut requested_frame, last, video_loaded);
        };
        let mut pick_request = None;
        let mut fitted_render_size = self.fitted_render_size;
        let mut view = crate::ui::View {
            controller: &mut self.controller,
            texture: self
                .shared
                .video_loaded
                .get()
                .then_some(self.display_texture.as_ref())
                .flatten(),
            render_size: self.display_size,
            last_render_ms: None,
            pick_request: &mut pick_request,
        };
        let mut extensions = crate::ui::UiExtensions {
            top_controls: Some(&mut top_controls),
            extra_controls: Some(&mut extra_controls),
            image_overlay: None,
            camera_locked: true,
            image_sizing: self.image_sizing,
            move_reference_labels: Some(["e1", "e2", "e3"]),
            hide_empty_image: true,
            fitted_render_size: Some(&mut fitted_render_size),
            central_bottom: Some(&mut central_bottom),
            central_bottom_height: Some(80.0),
        };
        let changed = crate::ui::show_with_extensions(ui, &mut view, &mut extensions);
        self.details_open = details_open;
        if open_video_requested {
            self.show_video_source_dialog = true;
            ui.ctx().request_repaint();
        }
        if let Some(command) = playback_command {
            self.shared.command.set(command);
        }
        if requested_frame != self.current_frame_index {
            self.current_frame_index = requested_frame;
            self.pending_seek_target = Some(requested_frame);
            self.shared.seek_frame.set(requested_frame as i32);
        }
        let fitted_render_size = (
            fitted_render_size.0.min(video.width).max(1),
            fitted_render_size.1.min(video.height).max(1),
        );
        if self.image_sizing == crate::ui::ImageSizing::FitCanvas
            && fitted_render_size != self.fitted_render_size
        {
            self.fitted_render_size = fitted_render_size;
            if self.selected_asset.is_some() {
                self.shared.request_overlay();
                ui.ctx().request_repaint();
            }
        }

        if let Some(asset) = requested_asset {
            self.selected_asset = Some(asset);
            self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
            self.controller.state.selected = Some(0);
            self.controller.target = crate::interaction::InteractionTarget::Object;
            self.shared.renderer.borrow_mut().take();
            self.shared.asset_request.set(asset.code());
            self.shared.request_overlay();
        }

        if let Some((x, y)) = pick_request {
            let mut scene_changed = false;
            let clicked_quad = quad.is_some_and(|points| {
                let source = [
                    x as f32 * video.width as f32 / self.display_size.0 as f32,
                    y as f32 * video.height as f32 / self.display_size.1 as f32,
                ];
                point_in_quad(source, points)
            });
            if self.shared.video_playing.get() {
                if self.selected_asset.is_some() && self.selected_quad {
                    self.shared.request_pick((x, y));
                }
            } else if self.selected_asset.is_some() && self.selected_quad {
                if clicked_quad && !self.show_quad_gizmo {
                    self.show_quad_gizmo = true;
                    scene_changed = true;
                } else {
                    self.shared.request_pick((x, y));
                }
            } else {
                self.selected_quad = clicked_quad;
                self.show_quad_gizmo = clicked_quad;
                scene_changed = true;
            }
            if self.selected_quad {
                self.controller.target = crate::interaction::InteractionTarget::Object;
            }
            if scene_changed {
                self.shared.request_overlay();
            }
        }

        if changed {
            self.shared.request_overlay();
            ui.ctx().request_repaint();
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
