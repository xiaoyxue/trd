//! The editor's own egui surface (#163/#167): the left-pane editing controls,
//! the quad/catalog interaction wiring, and the player footer.
//!
//! This is the *editing* UI. The read-only Details inspector lives in
//! [`super::details_ui`], and the shared viewer controls (Interaction,
//! Transform, Render mode, PBR material, Overlays, Selection) stay in
//! [`crate::ui`]; this module composes them for the video editor and owns the
//! playback widgets.

use super::details_ui::details_ui;
use super::{
    point_in_quad, CatalogAsset, VideoEditingApp, VideoSourceKind, COMMAND_PAUSE, COMMAND_PLAY,
};

impl eframe::App for VideoEditingApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.shared.context.replace(Some(ui.ctx().clone()));
        self.sync_native_texture(_frame);
        if let Some(video) = self.shared.take_pending_video_info() {
            // A timeline probed from the container after start-up (#264).
            self.set_video_info(video);
        }
        if let Some(document) = self.shared.take_incoming_document() {
            // A document attached, replaced or cleared from the Open dialog.
            self.set_document(document);
        }
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
        let quad = self
            .frame_row(overlay_frame_index)
            .and_then(|frame| frame.placement_quad);
        let quad_frame = self.quad_frame_at(overlay_frame_index);
        let mut needs_render = false;
        let mut pick = None;

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
                    needs_render |= crate::ui::controls_sections(
                        ui,
                        &mut self.controller,
                        crate::ui::Controls {
                            camera_locked: true,
                            move_reference_labels: Some(["e1", "e2", "e3"]),
                        },
                    );
                    ui.separator();
                    self.shot_controls(ui);
                    self.quad_controls(ui, overlay_frame_index, quad_frame);
                    self.catalog_controls(ui);
                    self.details_controls(ui);
                    needs_render |= crate::ui::reset_button(ui, &mut self.controller);
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
            // Field-level borrows: the panel takes `&mut self.controller`, so the
            // texture must be selected from disjoint fields rather than through a
            // `&self` method.
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
                    sizing: self.image_sizing,
                    camera_locked: true,
                    hide_when_empty: true,
                },
            );
            needs_render |= outcome.needs_render;
            if let Some(fitted) = outcome.fitted_size {
                self.resize_render_target(ui.ctx(), fitted);
            }
            pick = outcome.pick;
        });

        self.settle_frame(ui.ctx(), needs_render, pick, quad);
    }
}

impl VideoEditingApp {
    /// Applies one frame's [`ImageOutcome`](crate::ui::ImageOutcome) **in the
    /// order the pick contract requires**: the scene revision settles first, and
    /// only then does a pick request capture it.
    ///
    /// The order is the whole point of this function existing, so it is stated
    /// once here rather than inlined at the call site (#205).
    /// [`image_panel`](crate::ui::image_panel) reports `needs_render` for the
    /// *very same* primary click that requests the pick, and both
    /// [`accepts_pick`](VideoEditingShared::accepts_pick) and the
    /// [`schedule_pick`](Self::schedule_pick) completion guard reject a result
    /// whose request captured a stale revision. Handling the pick first —
    /// as it was until this was hoisted out of the `CentralPanel` closure —
    /// therefore made the click invalidate *its own* completion, and selection
    /// was dead. Bumping first is safe because
    /// [`handle_pick`](Self::handle_pick) never bumps the revision on the GPU
    /// pick path.
    pub(super) fn settle_frame(
        &mut self,
        ctx: &egui::Context,
        needs_render: bool,
        pick: Option<(u32, u32)>,
        quad: Option<[[f32; 2]; 4]>,
    ) {
        if needs_render {
            self.shared.request_overlay();
            ctx.request_repaint();
        }
        if let Some(point) = pick {
            self.handle_pick(point, quad);
        }
    }

    /// Source heading, the Open button, and the loaded-source readout.
    fn source_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Video");
        if ui.button("Open source...").clicked() {
            self.show_video_source_dialog = true;
            ui.ctx().request_repaint();
        }
        ui.weak("Display: fit right pane (16:9)");
        ui.collapsing("Source", |ui| {
            let video = &self.video;
            ui.label(format!("Source: {}", video.source_name));
            ui.label(match self.shared.pending_document() {
                Some(source) => match source.kind {
                    VideoSourceKind::LocalFile => format!("Document: {} (local)", source.name),
                    VideoSourceKind::HttpUrl => format!("Document: {}", source.name),
                },
                None => "Document: none — the video plays as-is".to_owned(),
            });
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
            if let Some(error) = self.shared.error.borrow().as_deref() {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
        });
    }

    /// The standalone placement-quad readout.
    fn quad_controls(
        &mut self,
        ui: &mut egui::Ui,
        frame_index: u32,
        quad_frame: Option<trd_placement::QuadFrame>,
    ) {
        ui.collapsing("Placement quad (standalone)", |ui| {
            let Some(frame) = self.frame_row(frame_index) else {
                // No document, or a frame it does not annotate: the video plays
                // and there is simply nothing to place here (#264).
                ui.label(format!("Frame {frame_index}"));
                ui.label(if self.has_document() {
                    "Not annotated: this frame is plain video"
                } else {
                    "No annotation document: the video plays as-is"
                });
                return;
            };
            ui.label(format!("Frame {}", frame.video_frame_index));
            ui.label(if frame.tracked {
                if self.shared.video_playing.get() {
                    "Placement quad hidden during playback"
                } else if !self.selected_quad {
                    "Click the green quad to select it"
                } else if self.show_quad_gizmo {
                    "Placement quad selected; gizmo visible"
                } else {
                    "Placement quad selected; click it to show gizmo"
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
    }

    /// The fixed catalog. Selecting an asset loads it immediately.
    fn catalog_controls(&mut self, ui: &mut egui::Ui) {
        ui.add_enabled_ui(self.selected_quad, |ui| {
            ui.collapsing("Object catalog", |ui| {
                for asset in CatalogAsset::ALL {
                    if ui
                        .selectable_label(self.selected_asset == Some(asset), asset.label())
                        .clicked()
                    {
                        self.select_catalog_asset(asset);
                        ui.ctx().request_repaint();
                    }
                }
            });
        });
    }

    /// The Details inspector. Its body only runs while expanded, so the facts
    /// are derived only when they are actually shown.
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

    /// The **Shots** section: where the annotated ranges are, and how to get to
    /// them.
    ///
    /// A sparse document may annotate a few hundred frames of a clip that runs
    /// to hundreds of thousands, so without this nothing tells a user *where*
    /// the editable parts are. Selecting a shot seeks to its **first** frame
    /// (#264).
    fn shot_controls(&mut self, ui: &mut egui::Ui) {
        let shots = self.shots();
        ui.collapsing(format!("Shots ({})", shots.len()), |ui| {
            // The overlay is an authoring aid, not something to watch a cut
            // through — so it is a toggle, and it governs playback too.
            ui.checkbox(&mut self.show_overlay, "Show placement overlay")
                .on_hover_text(
                    "Draw the quad and gizmo on annotated frames, including while playing",
                );
            if shots.is_empty() {
                ui.weak(if self.has_document() {
                    "The document annotates no frames"
                } else {
                    "No annotation document: the whole video is plain playback"
                });
                return;
            }
            let current = self.current_frame_index;
            for (index, shot) in shots.iter().enumerate() {
                let label = format!(
                    "Shot {} · frames {}-{} ({})",
                    index + 1,
                    shot.start_frame,
                    shot.end_frame,
                    shot.frame_count()
                );
                if ui
                    .selectable_label(shot.contains(current), label)
                    .on_hover_text("Jump to the first frame of this shot")
                    .clicked()
                {
                    self.seek_to(shot.start_frame);
                    ui.ctx().request_repaint();
                }
            }
        });
    }

    fn seek_to(&mut self, frame_index: u32) {
        if frame_index == self.current_frame_index {
            return;
        }
        self.current_frame_index = frame_index;
        self.pending_seek_target = Some(frame_index);
        self.shared.seek_frame.set(frame_index as i32);
    }

    fn select_catalog_asset(&mut self, asset: CatalogAsset) {
        self.selected_asset = Some(asset);
        self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
        self.controller.state.selected = Some(0);
        self.controller.target = crate::interaction::InteractionTarget::Object;
        self.shared.renderer.borrow_mut().take();
        self.shared.asset_request.set(asset.code());
        self.shared.request_overlay();
    }

    /// Adopts the letterboxed image size the panel just drew at.
    fn resize_render_target(&mut self, ctx: &egui::Context, fitted: (u32, u32)) {
        let video = &self.video;
        let fitted = (
            fitted.0.min(video.width).max(1),
            fitted.1.min(video.height).max(1),
        );
        if self.image_sizing != crate::ui::ImageSizing::FitCanvas
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

    /// Resolves a click on the image into a quad selection, a gizmo reveal, or a
    /// GPU pick request.
    pub(super) fn handle_pick(&mut self, (x, y): (u32, u32), quad: Option<[[f32; 2]; 4]>) {
        let video = &self.video;
        let clicked_quad = quad.is_some_and(|points| {
            let source = [
                x as f32 * video.width as f32 / self.display_size.0 as f32,
                y as f32 * video.height as f32 / self.display_size.1 as f32,
            ];
            point_in_quad(source, points)
        });
        let mut scene_changed = false;
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
