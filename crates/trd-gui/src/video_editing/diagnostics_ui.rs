//! egui rendering for the [`VideoEditingDiagnostics`](super::diagnostics)
//! snapshot (#167).
//!
//! Presentation only. Every value shown here is read straight off the immutable
//! snapshot; this module must not compute domain values of its own.

use super::diagnostics::VideoEditingDiagnostics;
use super::VideoSourceKind;

pub(super) fn video_editing_diagnostics_ui(
    ui: &mut egui::Ui,
    diagnostics: &VideoEditingDiagnostics,
) {
    ui.horizontal(|ui| {
        if ui.small_button("Copy diagnostics JSON").clicked() {
            match diagnostics.to_json() {
                Ok(json) => ui.ctx().copy_text(json),
                Err(error) => {
                    ui.colored_label(egui::Color32::LIGHT_RED, error.to_string());
                }
            }
        }
        ui.weak("Snapshot follows the displayed render.");
    });

    ui.collapsing("Source", |ui| {
        let source = &diagnostics.source;
        diagnostic_table(ui, "video-source-diagnostics", |ui| {
            diagnostic_row(
                ui,
                "kind",
                source
                    .observed_kind
                    .map(source_kind_label)
                    .unwrap_or("not loaded"),
            );
            diagnostic_match_row(
                ui,
                "name",
                source.expected_name.as_str(),
                source.observed_name.as_deref(),
            );
            diagnostic_match_row(
                ui,
                "byte length",
                source.expected_byte_length,
                source.observed_byte_length,
            );
            diagnostic_row(ui, "declared MIME", &source.expected_mime);
            diagnostic_row(ui, "declared codec", &source.expected_codec);
            diagnostic_match_row(
                ui,
                "dimensions",
                format!("{}x{}", source.expected_size[0], source.expected_size[1]),
                source
                    .observed_size
                    .map(|size| format!("{}x{}", size[0], size[1])),
            );
            diagnostic_row(
                ui,
                "FPS",
                format!("{}/{}", source.expected_fps[0], source.expected_fps[1]),
            );
            diagnostic_row(ui, "frame count", source.expected_frame_count);
            diagnostic_match_float_row(
                ui,
                "duration",
                source.expected_duration_seconds,
                source.observed_duration_seconds,
                f64::from(source.expected_fps[1]) / f64::from(source.expected_fps[0].max(1)),
                "s",
            );
            diagnostic_row(ui, "SHA-256", &source.expected_sha256);
            diagnostic_warning_row(ui, "digest", source.digest_status);
            diagnostic_row(ui, "media readyState", source.ready_state);
            diagnostic_row(ui, "loaded", source.loaded);
            diagnostic_row(ui, "playing", source.playing);
            diagnostic_row(ui, "ended", source.ended);
            diagnostic_optional_error_row(ui, "error", source.error.as_deref());
        });
    });

    ui.collapsing("Timeline / synchronization", |ui| {
        let timeline = &diagnostics.timeline;
        diagnostic_table(ui, "video-timeline-diagnostics", |ui| {
            diagnostic_row(
                ui,
                "media time",
                option_f64(timeline.media_time_seconds, "s"),
            );
            diagnostic_row(ui, "requested frame", timeline.requested_frame_index);
            diagnostic_row(
                ui,
                "presented frame",
                option_u32(timeline.presented_frame_index),
            );
            diagnostic_row(
                ui,
                "displayed frame",
                option_u32(timeline.displayed_frame_index),
            );
            diagnostic_row(
                ui,
                "rendered frame",
                option_u32(timeline.rendered_frame_index),
            );
            diagnostic_row(
                ui,
                "Arrow video_frame_index",
                option_u32(timeline.arrow_video_frame_index),
            );
            diagnostic_row(
                ui,
                "Arrow present_index",
                option_u32(timeline.present_index),
            );
            diagnostic_row(
                ui,
                "Arrow timestamp_us",
                timeline
                    .timestamp_us
                    .map_or_else(|| "none".to_owned(), |value| value.to_string()),
            );
            diagnostic_row(
                ui,
                "media/row delta",
                option_f64(timeline.media_timestamp_delta_ms, "ms"),
            );
            diagnostic_row(
                ui,
                "tracking state",
                timeline.tracked.map_or(
                    "none",
                    |tracked| {
                        if tracked {
                            "tracked"
                        } else {
                            "video-only"
                        }
                    },
                ),
            );
            diagnostic_row(
                ui,
                "source size",
                format!("{}x{}", timeline.source_size[0], timeline.source_size[1]),
            );
            diagnostic_row(
                ui,
                "render size",
                format!("{}x{}", timeline.render_size[0], timeline.render_size[1]),
            );
            diagnostic_row(ui, "source generation", timeline.source_generation);
            diagnostic_row(ui, "render revision", timeline.render_revision);
            diagnostic_row(
                ui,
                "pending render",
                option_u64(timeline.pending_render_generation),
            );
            diagnostic_row(
                ui,
                "in-flight frame",
                option_u32(timeline.in_flight_frame_index),
            );
            diagnostic_row(
                ui,
                "coalesced frame",
                option_u32(timeline.coalesced_frame_index),
            );
            diagnostic_row(
                ui,
                "last render",
                option_f64(timeline.last_render_latency_ms, "ms"),
            );
            diagnostic_row(
                ui,
                "average render",
                option_f64(timeline.average_render_latency_ms, "ms"),
            );
            diagnostic_row(ui, "seek target", option_u32(timeline.seek_target));
            diagnostic_row(ui, "seek pending", timeline.seek_pending);
        });
    });

    ui.collapsing("Tracking / quad frame", |ui| {
        let tracking = &diagnostics.tracking;
        diagnostic_table(ui, "video-tracking-diagnostics", |ui| {
            if let Some(points) = tracking.points_tl_tr_br_bl {
                for (label, point) in ["TL", "TR", "BR", "BL"].into_iter().zip(points) {
                    diagnostic_row(ui, label, vec2_label(point));
                }
            } else {
                diagnostic_row(ui, "quad points", "none");
            }
            if let Some([fx, fy, cx, cy]) = tracking.intrinsics_fx_fy_cx_cy {
                diagnostic_row(
                    ui,
                    "K (fx, fy, cx, cy)",
                    format!("{fx:.4}, {fy:.4}, {cx:.4}, {cy:.4}"),
                );
            } else {
                diagnostic_row(ui, "K", "none");
            }
            if let Some(frame) = &tracking.quad_frame {
                diagnostic_row(ui, "origin", vec3_label(frame.origin));
                diagnostic_row(ui, "e1", vec3_label(frame.e1));
                diagnostic_row(ui, "e2", vec3_label(frame.e2));
                diagnostic_row(ui, "e3", vec3_label(frame.e3));
                diagnostic_row(
                    ui,
                    "half-edge lengths",
                    format!(
                        "{:.6}, {:.6}",
                        frame.half_edge_lengths[0], frame.half_edge_lengths[1]
                    ),
                );
                diagnostic_row(ui, "axis length", format!("{:.6}", frame.axis_length));
                diagnostic_row(
                    ui,
                    "|dot(e1,e2/e3), dot(e2,e3)|",
                    format!(
                        "{:.6}, {:.6}, {:.6}",
                        frame.orthogonality_errors[0],
                        frame.orthogonality_errors[1],
                        frame.orthogonality_errors[2]
                    ),
                );
                diagnostic_row(
                    ui,
                    "handedness determinant",
                    format!("{:.6}", frame.handedness_determinant),
                );
            }
            if let Some(delta) = &tracking.pose_delta {
                diagnostic_row(ui, "previous tracked frame", delta.previous_frame_index);
                diagnostic_row(
                    ui,
                    "pose translation delta",
                    format!("{:.6}", delta.translation),
                );
                diagnostic_row(
                    ui,
                    "pose rotation delta",
                    format!("{:.4} deg", delta.rotation_degrees),
                );
                diagnostic_row(
                    ui,
                    "axis-length ratio",
                    format!("{:.6}", delta.axis_length_ratio),
                );
            } else {
                diagnostic_row(ui, "pose delta", "none");
            }
            if tracking.normal_sign_warning {
                diagnostic_warning_row(ui, "continuity", "quad normal sign changed");
            } else {
                diagnostic_row(ui, "continuity", "normal sign continuous");
            }
            if let Some(error) = tracking.placement_error {
                diagnostic_error_row(ui, "placement error", &error.to_string());
            } else {
                diagnostic_row(ui, "placement error", "none");
            }
            diagnostic_row(ui, "tracking smoothing", tracking.smoothing);
        });
    });

    ui.collapsing("Placement / object", |ui| {
        let placement = &diagnostics.placement;
        diagnostic_table(ui, "video-placement-diagnostics", |ui| {
            diagnostic_row(ui, "selected quad", placement.selected_quad);
            diagnostic_row(ui, "selected object", option_u32(placement.selected_object));
            diagnostic_row(ui, "catalog asset", option_label(placement.catalog_asset));
            diagnostic_row(ui, "source format", option_label(placement.source_format));
            diagnostic_row(
                ui,
                "preview AABB min",
                placement
                    .preview_aabb_min
                    .map_or_else(|| "none".to_owned(), vec3_label),
            );
            diagnostic_row(
                ui,
                "preview AABB max",
                placement
                    .preview_aabb_max
                    .map_or_else(|| "none".to_owned(), vec3_label),
            );
            diagnostic_row(
                ui,
                "preview scale",
                placement
                    .preview_scale
                    .map_or_else(|| "none".to_owned(), |value| format!("{value:.6}")),
            );
            diagnostic_row(
                ui,
                "Olympic preset",
                format!(
                    "size {:.2}, e1 {:.2}, e2 {:.2}, lift {:.2}",
                    placement.preset_size_factor,
                    placement.preset_offset_e1,
                    placement.preset_offset_e2,
                    placement.preset_lift
                ),
            );
            diagnostic_row(
                ui,
                "object translation",
                vec3_label(placement.object_translation),
            );
            diagnostic_row(
                ui,
                "object rotation",
                format!(
                    "yaw {:.3}, pitch {:.3}, roll {:.3} deg",
                    placement.object_rotation_degrees[0],
                    placement.object_rotation_degrees[1],
                    placement.object_rotation_degrees[2]
                ),
            );
            diagnostic_row(ui, "object scale", vec3_label(placement.object_scale));
            diagnostic_row(ui, "movement basis", placement.movement_basis.join(" / "));
            diagnostic_row(ui, "visibility", placement.visibility_reason);
        });
        if let Some(model) = diagnostics.placement.draw_model {
            ui.horizontal(|ui| {
                ui.label("draw_model");
                if ui.small_button("Copy").clicked() {
                    ui.ctx().copy_text(format_matrix(model));
                }
            });
            ui.monospace(format_matrix(model));
        } else {
            ui.weak("draw_model: none");
        }
    });

    ui.collapsing("Material / lighting", |ui| {
        let material = &diagnostics.material_lighting;
        diagnostic_table(ui, "video-material-diagnostics", |ui| {
            diagnostic_row(ui, "render mode", material.render_mode);
            diagnostic_row(
                ui,
                "imported metallic",
                option_f32(material.imported_metallic),
            );
            diagnostic_row(
                ui,
                "imported roughness",
                option_f32(material.imported_roughness),
            );
            diagnostic_row(ui, "base-color map", yes_no(material.base_color_map));
            diagnostic_row(
                ui,
                "metallic-roughness map",
                yes_no(material.metallic_roughness_map),
            );
            diagnostic_row(ui, "normal map", yes_no(material.normal_map));
            diagnostic_row(ui, "metallic", format!("{:.4}", material.metallic));
            diagnostic_row(ui, "roughness", format!("{:.4}", material.roughness));
            diagnostic_row(ui, "specular", format!("{:.4}", material.specular));
            diagnostic_row(ui, "clearcoat", format!("{:.4}", material.clearcoat));
            diagnostic_row(ui, "IBL", material.environment_name.unwrap_or("none"));
            diagnostic_row(
                ui,
                "IBL intensity / rotation",
                format!(
                    "{:.4} / {:.3} deg",
                    material.environment_intensity, material.environment_rotation_degrees
                ),
            );
            diagnostic_row(
                ui,
                "direct light / ambient",
                format!(
                    "{:.4} / {:.4}",
                    material.direct_light_scale, material.ambient
                ),
            );
            diagnostic_row(ui, "exposure", format!("{:.4}", material.exposure));
            diagnostic_row(ui, "tone map", material.tone_map);
            diagnostic_row(ui, "PBR debug", material.pbr_debug_view);
            if let Some(warning) = material.tracking_warning {
                diagnostic_warning_row(ui, "tracking/material", warning);
            }
        });
    });

    ui.collapsing("Renderer", |ui| {
        let renderer = &diagnostics.renderer;
        diagnostic_table(ui, "video-renderer-diagnostics", |ui| {
            diagnostic_row(
                ui,
                "adapter",
                option_label(renderer.adapter_name.as_deref()),
            );
            diagnostic_row(ui, "backend", option_label(renderer.backend.as_deref()));
            diagnostic_row(
                ui,
                "device type",
                option_label(renderer.device_type.as_deref()),
            );
            diagnostic_row(
                ui,
                "source size",
                format!("{}x{}", renderer.source_size[0], renderer.source_size[1]),
            );
            diagnostic_row(
                ui,
                "render target",
                format!(
                    "{}x{}",
                    renderer.render_target_size[0], renderer.render_target_size[1]
                ),
            );
            diagnostic_row(ui, "mode", renderer.mode);
            diagnostic_row(
                ui,
                "MSAA",
                renderer
                    .msaa_samples
                    .map_or_else(|| "unknown".to_owned(), |value| format!("{value}x")),
            );
            diagnostic_row(
                ui,
                "drawables (background/foreground/selection)",
                format!(
                    "{}/{}/{}",
                    renderer.background_drawables,
                    renderer.foreground_drawables,
                    renderer.selection_drawables
                ),
            );
            diagnostic_row(
                ui,
                "frame texture upload",
                renderer
                    .frame_texture_upload_bytes
                    .map_or_else(|| "none".to_owned(), |bytes| format!("{bytes} bytes")),
            );
            diagnostic_row(
                ui,
                "pick target",
                renderer.pick_target_size.map_or_else(
                    || "none".to_owned(),
                    |size| format!("{}x{}", size[0], size[1]),
                ),
            );
            diagnostic_row(
                ui,
                "latest pick",
                renderer.latest_pick_result.map_or_else(
                    || "none".to_owned(),
                    |hit| hit.map_or_else(|| "miss".to_owned(), |id| format!("object {id}")),
                ),
            );
            diagnostic_optional_error_row(
                ui,
                "last render error",
                renderer.last_render_error.as_deref(),
            );
            diagnostic_optional_error_row(
                ui,
                "last pick error",
                renderer.last_pick_error.as_deref(),
            );
        });
    });
}

fn diagnostic_table(ui: &mut egui::Ui, id: &'static str, add_rows: impl FnOnce(&mut egui::Ui)) {
    egui::Grid::new(id)
        .num_columns(3)
        .striped(true)
        .show(ui, add_rows);
}

fn diagnostic_row(ui: &mut egui::Ui, label: &str, value: impl ToString) {
    ui.label(label);
    ui.monospace(value.to_string());
    ui.label("");
    ui.end_row();
}

fn diagnostic_match_row<T>(ui: &mut egui::Ui, label: &str, expected: T, observed: Option<T>)
where
    T: PartialEq + ToString,
{
    ui.label(label);
    let expected_text = expected.to_string();
    let observed_text = observed.as_ref().map(ToString::to_string);
    ui.monospace(format!(
        "expected {expected_text} / observed {}",
        observed_text.as_deref().unwrap_or("none")
    ));
    match observed {
        Some(observed) if observed == expected => {
            ui.colored_label(egui::Color32::LIGHT_GREEN, "MATCH");
        }
        Some(_) => {
            ui.colored_label(egui::Color32::LIGHT_RED, "MISMATCH");
        }
        None => {
            ui.colored_label(egui::Color32::YELLOW, "NOT OBSERVED");
        }
    }
    ui.end_row();
}

fn diagnostic_match_float_row(
    ui: &mut egui::Ui,
    label: &str,
    expected: f64,
    observed: Option<f64>,
    tolerance: f64,
    unit: &str,
) {
    ui.label(label);
    ui.monospace(format!(
        "expected {expected:.6}{unit} / observed {}",
        observed.map_or_else(|| "none".to_owned(), |value| format!("{value:.6}{unit}"))
    ));
    match observed {
        Some(value) if (value - expected).abs() <= tolerance => {
            ui.colored_label(egui::Color32::LIGHT_GREEN, "MATCH");
        }
        Some(_) => {
            ui.colored_label(egui::Color32::LIGHT_RED, "MISMATCH");
        }
        None => {
            ui.colored_label(egui::Color32::YELLOW, "NOT OBSERVED");
        }
    }
    ui.end_row();
}

fn diagnostic_warning_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(label);
    ui.monospace(value);
    ui.colored_label(egui::Color32::YELLOW, "WARNING");
    ui.end_row();
}

fn diagnostic_error_row(ui: &mut egui::Ui, label: &str, value: &str) {
    ui.label(label);
    ui.monospace(value);
    ui.colored_label(egui::Color32::LIGHT_RED, "ERROR");
    ui.end_row();
}

fn diagnostic_optional_error_row(ui: &mut egui::Ui, label: &str, value: Option<&str>) {
    if let Some(value) = value {
        diagnostic_error_row(ui, label, value);
    } else {
        diagnostic_row(ui, label, "none");
    }
}

fn source_kind_label(kind: VideoSourceKind) -> &'static str {
    match kind {
        VideoSourceKind::LocalFile => "local file",
        VideoSourceKind::HttpUrl => "HTTP(S) URL",
    }
}

fn option_label(value: Option<&str>) -> &str {
    value.unwrap_or("none")
}

fn option_u32(value: Option<u32>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

fn option_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| value.to_string())
}

fn option_f32(value: Option<f32>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("{value:.6}"))
}

fn option_f64(value: Option<f64>, unit: &str) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("{value:.3} {unit}"))
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn vec2_label(value: [f32; 2]) -> String {
    format!("[{:.4}, {:.4}]", value[0], value[1])
}

fn vec3_label(value: [f32; 3]) -> String {
    format!("[{:.6}, {:.6}, {:.6}]", value[0], value[1], value[2])
}

fn format_matrix(matrix: [f32; 16]) -> String {
    (0..4)
        .map(|row| {
            format!(
                "[{:.6}, {:.6}, {:.6}, {:.6}]",
                matrix[row],
                matrix[4 + row],
                matrix[8 + row],
                matrix[12 + row]
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
