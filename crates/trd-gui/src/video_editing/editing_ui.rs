//! The editor's egui surface (#163/#167): left-pane editing controls,
//! quad/catalog interaction, and player footer.

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
            self.set_document(document);
        }
        if let Some(scene) = self.shared.take_incoming_scene() {
            self.set_arrow_scene(scene);
        }
        self.consume_video_frame();
        self.consume_rendered_frame();
        self.consume_asset_defaults();
        self.consume_pick_result();
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
        let quad = self
            .frame_row(overlay_frame_index)
            .and_then(|frame| frame.placement_quad);
        let quad_frame = self.quad_frame_at(overlay_frame_index);
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
            hover = outcome.hover;
            if let Some(rect) = outcome.image_rect {
                self.paint_axis_labels(ui, rect, overlay_frame_index, quad_frame);
            }
        });

        self.settle_frame(ui.ctx(), needs_render, pick, hover, quad);
    }
}

impl VideoEditingApp {
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
            let source = [
                x as f32 * self.video.width as f32 / self.display_size.0 as f32,
                y as f32 * self.video.height as f32 / self.display_size.1 as f32,
            ];
            point_in_quad(source, points)
        })
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
                    VideoSourceKind::LocalFile => format!("Arrow input: {} (local)", source.name),
                    VideoSourceKind::HttpUrl => format!("Arrow input: {}", source.name),
                },
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
                    ui.weak("Replay mode: the exported models are rendered over this video.");
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

    /// The standalone placement-quad readout.
    fn quad_controls(
        &mut self,
        ui: &mut egui::Ui,
        frame_index: u32,
        quad_frame: Option<trd_placement::QuadFrame>,
    ) {
        ui.collapsing("Placement quad (standalone)", |ui| {
            let Some(frame) = self.frame_row(frame_index) else {
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
                if self.shared.video_playing.get() && !self.show_placement_quads {
                    "Placement quad hidden during playback"
                } else if !self.selected_quad {
                    "Click the green quad to select it"
                } else {
                    "Placement quad selected"
                }
            } else {
                "Background-only row: quad and object hidden"
            });
            ui.weak(format!(
                "Pointer: {} · quad: {}",
                if self.hovered_quad {
                    "over the quad"
                } else {
                    "off the quad"
                },
                if self.selected_quad {
                    "selected"
                } else {
                    "not selected"
                }
            ));
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
        })
        .response
        .on_disabled_hover_text("Select a placement quad first");
    }

    fn export_controls(&self, ui: &mut egui::Ui) {
        ui.collapsing("Scene export", |ui| {
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
                    ui.weak("Protocol 0.0.6 scene; the source video remains a sidecar.");
                }
            }
            ui.weak("PBR export keeps geometry and base color, not material maps or IBL.");
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

    /// Shots section: click a shot to seek to its first frame (#264).
    fn shot_controls(&mut self, ui: &mut egui::Ui) {
        let shots = self.shots();
        ui.collapsing(format!("Shots ({})", shots.len()), |ui| {
            let mut changed = ui
                .checkbox(&mut self.show_placement_quads, "Show placement quads")
                .on_hover_text(
                    "Draw the placement quad on annotated frames, including while playing",
                )
                .changed();
            changed |= ui
                .checkbox(&mut self.show_gizmos, "Show gizmos")
                .on_hover_text("Draw the quad's local floor grid and basis axes (e1 / e2 / e3)")
                .changed();
            if changed {
                self.shared.request_overlay();
            }
            let state = super::overlay_state(
                self.show_placement_quads || self.show_gizmos,
                self.has_document(),
                self.current_frame_index,
                self.frame_row(self.current_frame_index)
                    .map(|frame| frame.tracked),
            );
            ui.weak(state.label());
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
        // Seek id retires the request when it arrives back (#322).
        self.pending_seek = Some(super::PendingSeek {
            frame_index,
            id: self.shared.request_seek(frame_index),
        });
    }

    fn select_catalog_asset(&mut self, asset: CatalogAsset) {
        self.selected_asset = Some(asset);
        self.selected_quad = true;
        self.show_gizmos = true;
        self.controller.state.objects[0] = crate::scene::ObjectTransform::default();
        self.controller.state.selected = Some(0);
        self.controller.target = crate::interaction::InteractionTarget::Object;
        self.shared.clear_export_asset();
        self.shared.cancel_arrow_export();
        self.shared.renderer.borrow_mut().take();
        self.shared.asset_request.set(asset.code());
        self.shared.request_overlay();
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
        let Some(k) = self.frame_row(frame_index).and_then(|row| row.k) else {
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

    /// Resolves a click on the image into a quad selection or a GPU pick request.
    ///
    /// Resolves a click into a quad selection or GPU pick. Selection bumps before
    /// pick so the pick captures the updated revision (#205).
    pub(super) fn handle_pick(&mut self, (x, y): (u32, u32), quad: Option<[[f32; 2]; 4]>) {
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
