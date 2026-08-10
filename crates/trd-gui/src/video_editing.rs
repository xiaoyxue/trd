//! Shared browser/native video-editing state and UI (#163/#167).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CatalogAsset {
    CocaColaCan,
    BeerCan,
    Dragon,
}

impl CatalogAsset {
    pub const ALL: [Self; 3] = [Self::CocaColaCan, Self::BeerCan, Self::Dragon];

    pub const fn label(self) -> &'static str {
        match self {
            Self::CocaColaCan => "Coca-Cola can",
            Self::BeerCan => "Beer can",
            Self::Dragon => "Dragon",
        }
    }
}

fn video_editing_diagnostics_ui(ui: &mut egui::Ui, diagnostics: &VideoEditingDiagnostics) {
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

impl CatalogAsset {
    pub const fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::CocaColaCan),
            2 => Some(Self::BeerCan),
            3 => Some(Self::Dragon),
            _ => None,
        }
    }

    pub const fn code(self) -> u8 {
        self as u8 + 1
    }
}

const COMMAND_NONE: u8 = 0;
const COMMAND_PICK_VIDEO: u8 = 1;
const COMMAND_PLAY: u8 = 2;
const COMMAND_PAUSE: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoEditingCommand {
    OpenLocalVideo,
    Play,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoSourceKind {
    LocalFile,
    HttpUrl,
}

#[derive(Debug, Clone, PartialEq)]
struct VideoSourceObservation {
    kind: VideoSourceKind,
    name: String,
    byte_length: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct VideoMetadataObservation {
    width: u32,
    height: u32,
    duration_seconds: f64,
}

/// Media-element level state. It is deliberately *not* per-frame: `mediaTime`
/// travels with its own frame (`IncomingVideoFrame::media_time_seconds`) so the
/// timeline diagnostics describe the frame on screen, not a newer one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct VideoMediaObservation {
    ready_state: u8,
    ended: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct VideoEditingDiagnostics {
    pub source: SourceDiagnostics,
    pub timeline: TimelineDiagnostics,
    pub tracking: TrackingDiagnostics,
    pub placement: PlacementDiagnostics,
    pub material_lighting: MaterialLightingDiagnostics,
    pub renderer: RendererDiagnostics,
}

impl VideoEditingDiagnostics {
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SourceDiagnostics {
    pub expected_name: String,
    pub expected_byte_length: u64,
    pub expected_mime: String,
    pub expected_codec: String,
    pub expected_size: [u32; 2],
    pub expected_fps: [u32; 2],
    pub expected_frame_count: u32,
    pub expected_duration_seconds: f64,
    pub expected_sha256: String,
    pub observed_kind: Option<VideoSourceKind>,
    pub observed_name: Option<String>,
    pub observed_byte_length: Option<u64>,
    pub observed_size: Option<[u32; 2]>,
    pub observed_duration_seconds: Option<f64>,
    pub ready_state: u8,
    pub loaded: bool,
    pub playing: bool,
    pub ended: bool,
    pub error: Option<String>,
    pub digest_status: &'static str,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TimelineDiagnostics {
    pub media_time_seconds: Option<f64>,
    pub requested_frame_index: u32,
    pub presented_frame_index: Option<u32>,
    pub displayed_frame_index: Option<u32>,
    pub rendered_frame_index: Option<u32>,
    pub arrow_video_frame_index: Option<u32>,
    pub present_index: Option<u32>,
    pub timestamp_us: Option<i64>,
    pub media_timestamp_delta_ms: Option<f64>,
    pub tracked: Option<bool>,
    pub source_size: [u32; 2],
    pub render_size: [u32; 2],
    pub source_generation: u64,
    pub render_revision: u64,
    pub pending_render_generation: Option<u64>,
    pub in_flight_frame_index: Option<u32>,
    pub coalesced_frame_index: Option<u32>,
    pub last_render_latency_ms: Option<f64>,
    pub average_render_latency_ms: Option<f64>,
    pub seek_target: Option<u32>,
    pub seek_pending: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TrackingDiagnostics {
    pub points_tl_tr_br_bl: Option<[[f32; 2]; 4]>,
    pub intrinsics_fx_fy_cx_cy: Option<[f32; 4]>,
    pub quad_frame: Option<QuadFrameDiagnostics>,
    pub pose_delta: Option<PoseDeltaDiagnostics>,
    pub normal_sign_warning: bool,
    pub placement_error: Option<TrackingPlacementError>,
    pub smoothing: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum TrackingPlacementError {
    #[error("frame is outside the timeline")]
    FrameOutOfRange,
    #[error("tracking row has no camera intrinsics")]
    MissingIntrinsics,
    #[error("tracking row has no placement quad")]
    MissingQuad,
    #[error("quad is degenerate")]
    DegenerateQuad,
    #[error("camera intrinsics are singular")]
    SingularIntrinsics,
    #[error("quad resolves behind the camera")]
    BehindCamera,
    #[error("placement scale must be positive")]
    InvalidScale,
}

impl From<trd_placement::PlacementError> for TrackingPlacementError {
    fn from(error: trd_placement::PlacementError) -> Self {
        match error {
            trd_placement::PlacementError::DegenerateQuad => Self::DegenerateQuad,
            trd_placement::PlacementError::SingularIntrinsics => Self::SingularIntrinsics,
            trd_placement::PlacementError::BehindCamera => Self::BehindCamera,
            trd_placement::PlacementError::InvalidScale => Self::InvalidScale,
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct QuadFrameDiagnostics {
    pub origin: [f32; 3],
    pub e1: [f32; 3],
    pub e2: [f32; 3],
    pub e3: [f32; 3],
    pub half_edge_lengths: [f32; 2],
    pub axis_length: f32,
    pub orthogonality_errors: [f32; 3],
    pub handedness_determinant: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PoseDeltaDiagnostics {
    pub previous_frame_index: u32,
    pub translation: f32,
    pub rotation_degrees: f32,
    pub axis_length_ratio: f32,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct PlacementDiagnostics {
    pub selected_quad: bool,
    pub selected_object: Option<u32>,
    pub catalog_asset: Option<&'static str>,
    pub source_format: Option<&'static str>,
    pub preview_aabb_min: Option<[f32; 3]>,
    pub preview_aabb_max: Option<[f32; 3]>,
    pub preview_scale: Option<f32>,
    pub preset_size_factor: f32,
    pub preset_offset_e1: f32,
    pub preset_offset_e2: f32,
    pub preset_lift: f32,
    pub object_translation: [f32; 3],
    pub object_rotation_degrees: [f32; 3],
    pub object_scale: [f32; 3],
    pub movement_basis: [&'static str; 3],
    pub draw_model: Option<[f32; 16]>,
    pub visibility_reason: &'static str,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MaterialLightingDiagnostics {
    pub render_mode: &'static str,
    pub imported_metallic: Option<f32>,
    pub imported_roughness: Option<f32>,
    pub base_color_map: bool,
    pub metallic_roughness_map: bool,
    pub normal_map: bool,
    pub metallic: f32,
    pub roughness: f32,
    pub specular: f32,
    pub clearcoat: f32,
    pub environment_name: Option<&'static str>,
    pub environment_intensity: f32,
    pub environment_rotation_degrees: f32,
    pub direct_light_scale: f32,
    pub ambient: f32,
    pub exposure: f32,
    pub tone_map: &'static str,
    pub pbr_debug_view: &'static str,
    pub tracking_warning: Option<&'static str>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct RendererDiagnostics {
    pub adapter_name: Option<String>,
    pub backend: Option<String>,
    pub device_type: Option<String>,
    pub source_size: [u32; 2],
    pub render_target_size: [u32; 2],
    pub mode: &'static str,
    pub msaa_samples: Option<u32>,
    pub background_drawables: u32,
    pub foreground_drawables: u32,
    pub selection_drawables: u32,
    pub frame_texture_upload_bytes: Option<u64>,
    pub pick_target_size: Option<[u32; 2]>,
    pub latest_pick_result: Option<Option<u32>>,
    pub last_render_error: Option<String>,
    pub last_pick_error: Option<String>,
}

#[derive(Clone)]
struct IncomingVideoFrame {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
    frame_index: u32,
    media_time_seconds: f64,
    source_generation: u64,
}

struct RenderedVideoFrame {
    frame: IncomingVideoFrame,
    render_revision: u64,
    diagnostics: RenderedFrameDiagnostics,
}

#[derive(Clone)]
struct RenderedFrameDiagnostics {
    media_time_seconds: f64,
    scene: crate::scene::SceneState,
    selected_asset: Option<CatalogAsset>,
    selected_quad: bool,
    move_direction: crate::interaction::MoveDirection,
    playing: bool,
    show_quad: bool,
    show_quad_gizmo: bool,
    draw_model: Option<trd_core::Matrix4>,
    renderer: crate::video_editing_renderer::VideoRendererDiagnostics,
}

#[derive(Clone, Copy)]
struct PickRequest {
    id: u64,
    point: (u32, u32),
    source_generation: u64,
    render_revision: u64,
}

struct PickResult {
    id: u64,
    source_generation: u64,
    render_revision: u64,
    hit: Option<u32>,
}

pub struct VideoEditingShared {
    frame: RefCell<Option<IncomingVideoFrame>>,
    latest_video_frame: RefCell<Option<IncomingVideoFrame>>,
    rendered_frame: RefCell<Option<RenderedVideoFrame>>,
    context: RefCell<Option<egui::Context>>,
    command: Cell<u8>,
    asset_request: Cell<u8>,
    video_url_request: RefCell<Option<String>>,
    seek_frame: Cell<i32>,
    video_loaded: Cell<bool>,
    video_playing: Cell<bool>,
    video_source: RefCell<Option<VideoSourceObservation>>,
    video_metadata: Cell<Option<VideoMetadataObservation>>,
    video_media: Cell<VideoMediaObservation>,
    source_generation: Cell<u64>,
    needs_overlay: Cell<bool>,
    render_revision: Cell<u64>,
    render_in_flight: Cell<bool>,
    render_in_flight_frame: Cell<Option<u32>>,
    last_render_latency_ms: Cell<Option<f64>>,
    render_latency_total_ms: Cell<f64>,
    render_latency_count: Cell<u64>,
    last_render_error: RefCell<Option<String>>,
    pending_pick: Cell<Option<PickRequest>>,
    pick_revision: Cell<u64>,
    pick_in_flight: Cell<bool>,
    pick_result: RefCell<Option<PickResult>>,
    last_pick_error: RefCell<Option<String>>,
    renderer_generation: Cell<u64>,
    renderer: RefCell<Option<crate::video_editing_renderer::VideoPlacementRenderer>>,
    renderer_diagnostics: RefCell<Option<crate::video_editing_renderer::VideoRendererDiagnostics>>,
    asset_defaults: RefCell<Option<(CatalogAsset, trd_core::RenderMode, trd_core::DisneyMaterial)>>,
    error: RefCell<Option<String>>,
}

impl Default for VideoEditingShared {
    fn default() -> Self {
        Self {
            frame: RefCell::new(None),
            latest_video_frame: RefCell::new(None),
            rendered_frame: RefCell::new(None),
            context: RefCell::new(None),
            command: Cell::new(COMMAND_NONE),
            asset_request: Cell::new(0),
            video_url_request: RefCell::new(None),
            seek_frame: Cell::new(-1),
            video_loaded: Cell::new(false),
            video_playing: Cell::new(false),
            video_source: RefCell::new(None),
            video_metadata: Cell::new(None),
            video_media: Cell::new(VideoMediaObservation::default()),
            source_generation: Cell::new(0),
            needs_overlay: Cell::new(false),
            render_revision: Cell::new(0),
            render_in_flight: Cell::new(false),
            render_in_flight_frame: Cell::new(None),
            last_render_latency_ms: Cell::new(None),
            render_latency_total_ms: Cell::new(0.0),
            render_latency_count: Cell::new(0),
            last_render_error: RefCell::new(None),
            pending_pick: Cell::new(None),
            pick_revision: Cell::new(0),
            pick_in_flight: Cell::new(false),
            pick_result: RefCell::new(None),
            last_pick_error: RefCell::new(None),
            renderer_generation: Cell::new(0),
            renderer: RefCell::new(None),
            renderer_diagnostics: RefCell::new(None),
            asset_defaults: RefCell::new(None),
            error: RefCell::new(None),
        }
    }
}

impl VideoEditingShared {
    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
    ) -> Result<(), String> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(format!(
                "video RGBA length {} != {width}x{height}x4",
                rgba.len()
            ));
        }
        self.frame.replace(Some(IncomingVideoFrame {
            rgba,
            width,
            height,
            frame_index,
            media_time_seconds,
            source_generation: self.source_generation.get(),
        }));
        self.request_repaint();
        Ok(())
    }

    pub fn set_video_status(&self, loaded: bool, playing: bool) {
        if !loaded {
            self.source_generation
                .set(self.source_generation.get().wrapping_add(1));
            self.frame.replace(None);
            self.latest_video_frame.replace(None);
            self.rendered_frame.replace(None);
            self.pending_pick.set(None);
            self.pick_result.replace(None);
            self.last_pick_error.replace(None);
            self.last_render_error.replace(None);
            self.needs_overlay.set(false);
            self.render_in_flight_frame.set(None);
            self.video_media.set(VideoMediaObservation::default());
        }
        self.video_loaded.set(loaded);
        self.video_playing.set(playing);
        if !loaded {
            self.error.replace(None);
        }
        self.request_repaint();
    }

    pub fn set_video_source_observation(
        &self,
        kind: VideoSourceKind,
        name: impl Into<String>,
        byte_length: Option<u64>,
    ) {
        self.video_metadata.set(None);
        self.video_media.set(VideoMediaObservation::default());
        self.video_source.replace(Some(VideoSourceObservation {
            kind,
            name: name.into(),
            byte_length,
        }));
        self.request_repaint();
    }

    pub fn set_video_metadata_observation(&self, width: u32, height: u32, duration_seconds: f64) {
        self.video_metadata.set(Some(VideoMetadataObservation {
            width,
            height,
            duration_seconds,
        }));
        self.request_repaint();
    }

    pub fn set_video_media_observation(&self, ready_state: u8, ended: bool) {
        self.video_media
            .set(VideoMediaObservation { ready_state, ended });
        self.request_repaint();
    }

    pub fn set_error(&self, message: impl Into<String>) {
        self.error.replace(Some(message.into()));
        self.request_repaint();
    }

    pub fn clear_error(&self) {
        self.error.replace(None);
    }

    pub fn take_command(&self) -> Option<VideoEditingCommand> {
        match self.command.replace(COMMAND_NONE) {
            COMMAND_PICK_VIDEO => Some(VideoEditingCommand::OpenLocalVideo),
            COMMAND_PLAY => Some(VideoEditingCommand::Play),
            COMMAND_PAUSE => Some(VideoEditingCommand::Pause),
            _ => None,
        }
    }

    pub fn take_asset_request(&self) -> Option<CatalogAsset> {
        CatalogAsset::from_code(self.asset_request.replace(0))
    }

    pub fn take_video_url_request(&self) -> Option<String> {
        self.video_url_request.borrow_mut().take()
    }

    pub fn take_seek_frame(&self) -> Option<u32> {
        let frame = self.seek_frame.replace(-1);
        (frame >= 0).then_some(frame as u32)
    }

    pub fn set_renderer(&self, renderer: crate::video_editing_renderer::VideoPlacementRenderer) {
        self.renderer_generation
            .set(self.renderer_generation.get().wrapping_add(1));
        self.renderer_diagnostics
            .replace(Some(renderer.diagnostics()));
        self.renderer.replace(Some(renderer));
        self.request_overlay();
        self.request_repaint();
    }

    pub fn set_catalog_renderer(
        &self,
        asset: CatalogAsset,
        renderer: crate::video_editing_renderer::VideoPlacementRenderer,
    ) {
        let (mode, material) = renderer.defaults();
        self.asset_defaults.replace(Some((asset, mode, material)));
        self.set_renderer(renderer);
    }

    pub fn request_repaint(&self) {
        if let Some(context) = self.context.borrow().as_ref() {
            context.request_repaint();
        }
    }

    fn request_overlay(&self) {
        self.render_revision
            .set(self.render_revision.get().wrapping_add(1));
        self.needs_overlay.set(true);
    }

    fn request_pick(&self, point: (u32, u32)) {
        let id = self.pick_revision.get().wrapping_add(1);
        self.pick_revision.set(id);
        self.pending_pick.set(Some(PickRequest {
            id,
            point,
            source_generation: self.source_generation.get(),
            render_revision: self.render_revision.get(),
        }));
    }

    fn accepts_render(&self, rendered: &RenderedVideoFrame) -> bool {
        rendered.frame.source_generation == self.source_generation.get()
            && rendered.render_revision == self.render_revision.get()
    }

    fn accepts_pick(&self, result: &PickResult) -> bool {
        result.id == self.pick_revision.get()
            && result.source_generation == self.source_generation.get()
            && result.render_revision == self.render_revision.get()
    }

    fn record_render_latency(&self, started: Instant) {
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
        self.last_render_latency_ms.set(Some(latency_ms));
        self.render_latency_total_ms
            .set(self.render_latency_total_ms.get() + latency_ms);
        self.render_latency_count
            .set(self.render_latency_count.get().saturating_add(1));
    }
}

/// Browser bridge for the dedicated editor. It transfers browser-decoded pixels
/// and services commands emitted by Rust UI; it never computes scene matrices.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub struct VideoEditingHandle {
    shared: Rc<VideoEditingShared>,
    source_name: String,
    byte_length: u64,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
    width: u32,
    height: u32,
}

#[cfg(target_arch = "wasm32")]
impl VideoEditingHandle {
    pub(crate) fn new(
        document: &trd_core::VideoEditingDocument,
        shared: Rc<VideoEditingShared>,
    ) -> Self {
        Self {
            shared,
            source_name: document.video.source_name.clone(),
            byte_length: document.video.byte_length,
            fps_num: document.video.fps_num,
            fps_den: document.video.fps_den,
            frame_count: document.video.frame_count,
            width: document.video.width,
            height: document.video.height,
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
impl VideoEditingHandle {
    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoFile)]
    pub fn validate_video_file(
        &self,
        filename: &str,
        byte_length: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if filename != self.source_name {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}, got {filename}",
                self.source_name
            )));
        }
        if byte_length != self.byte_length as f64 {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {} bytes, got {byte_length:.0}",
                self.byte_length
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = validateVideoMetadata)]
    pub fn validate_video_metadata(
        &self,
        width: u32,
        height: u32,
        duration_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        self.shared
            .set_video_metadata_observation(width, height, duration_seconds);
        if (width, height) != (self.width, self.height) {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {}x{} video, got {width}x{height}",
                self.width, self.height
            )));
        }
        let expected_duration =
            f64::from(self.frame_count) * f64::from(self.fps_den) / f64::from(self.fps_num);
        let frame_duration = f64::from(self.fps_den) / f64::from(self.fps_num);
        if !duration_seconds.is_finite()
            || (duration_seconds - expected_duration).abs() > frame_duration
        {
            return Err(wasm_bindgen::JsValue::from_str(&format!(
                "expected {expected_duration:.3}s video, got {duration_seconds:.3}s"
            )));
        }
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = frameIndexAtMediaTime)]
    pub fn frame_index_at_media_time(&self, media_time_seconds: f64) -> u32 {
        frame_index_at_media_time(
            media_time_seconds,
            self.fps_num,
            self.fps_den,
            self.frame_count,
        )
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = mediaTimeAtFrame)]
    pub fn media_time_at_frame(&self, frame_index: u32) -> f64 {
        media_time_at_frame(frame_index, self.fps_num, self.fps_den, self.frame_count)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = updateVideoFrameRgba)]
    pub fn update_video_frame_rgba(
        &self,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        frame_index: u32,
        media_time_seconds: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        if frame_index >= self.frame_count {
            return Err(wasm_bindgen::JsValue::from_str(
                "video frame index out of range",
            ));
        }
        self.shared
            .update_video_frame_rgba(rgba, width, height, frame_index, media_time_seconds)
            .map_err(|error| wasm_bindgen::JsValue::from_str(&error))
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoStatus)]
    pub fn set_video_status(&self, loaded: bool, playing: bool) {
        self.shared.set_video_status(loaded, playing);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoSourceInfo)]
    pub fn set_video_source_info(
        &self,
        source_kind: u8,
        name: String,
        byte_length: f64,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let kind = match source_kind {
            1 => VideoSourceKind::LocalFile,
            2 => VideoSourceKind::HttpUrl,
            _ => return Err(wasm_bindgen::JsValue::from_str("unknown video source kind")),
        };
        let byte_length = (byte_length >= 0.0).then_some(byte_length as u64);
        self.shared
            .set_video_source_observation(kind, name, byte_length);
        Ok(())
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoMediaState)]
    pub fn set_video_media_state(&self, ready_state: u8, ended: bool) {
        self.shared.set_video_media_observation(ready_state, ended);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = setVideoError)]
    pub fn set_video_error(&self, message: String) {
        self.shared.set_error(message);
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeCommand)]
    pub fn take_command(&self) -> u8 {
        self.shared.command.replace(COMMAND_NONE)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeAssetRequest)]
    pub fn take_asset_request(&self) -> u8 {
        self.shared.asset_request.replace(0)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeVideoUrlRequest)]
    pub fn take_video_url_request(&self) -> Option<String> {
        self.shared.video_url_request.borrow_mut().take()
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = takeSeekFrame)]
    pub fn take_seek_frame(&self) -> i32 {
        self.shared.seek_frame.replace(-1)
    }

    #[wasm_bindgen::prelude::wasm_bindgen(js_name = loadCatalogAsset)]
    pub async fn load_catalog_asset(
        &self,
        asset_code: u8,
        model_bytes: Vec<u8>,
        texture_bytes: Vec<u8>,
        env_bytes: Vec<u8>,
    ) -> Result<(), wasm_bindgen::JsValue> {
        let asset = CatalogAsset::from_code(asset_code)
            .ok_or_else(|| wasm_bindgen::JsValue::from_str("unknown catalog asset"))?;
        let renderer = crate::video_editing_renderer::VideoPlacementRenderer::new(
            asset,
            &model_bytes,
            &texture_bytes,
            &env_bytes,
            self.width,
            self.height,
        )
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error))?;
        self.shared.set_catalog_renderer(asset, renderer);
        Ok(())
    }
}

pub struct VideoEditingApp {
    document: trd_core::VideoEditingDocument,
    display_image: egui::ColorImage,
    display_texture: Option<egui::TextureHandle>,
    current_frame_index: u32,
    displayed_frame_index: u32,
    displayed_frame_ready: bool,
    last_rendered_frame_index: Option<u32>,
    displayed_diagnostics: Option<RenderedFrameDiagnostics>,
    display_size: (u32, u32),
    shared: Rc<VideoEditingShared>,
    controller: crate::interaction::InteractionController,
    selected_quad: bool,
    show_quad_gizmo: bool,
    was_playing: bool,
    selected_asset: Option<CatalogAsset>,
    image_sizing: crate::ui::ImageSizing,
    fitted_render_size: (u32, u32),
    show_video_source_dialog: bool,
    video_url: String,
    pending_seek_target: Option<u32>,
    last_pick_result: Option<Option<u32>>,
    details_open: bool,
}

impl VideoEditingApp {
    pub fn new(document: trd_core::VideoEditingDocument, shared: Rc<VideoEditingShared>) -> Self {
        let source_size = (document.video.width, document.video.height);
        let scene = crate::scene::SceneState::default();
        let mut controller = crate::interaction::InteractionController::new(scene);
        controller.target = crate::interaction::InteractionTarget::Object;
        controller.move_direction = crate::interaction::MoveDirection::Reference1;
        controller.move_reference_axes = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
        controller.state.camera.distance = 1.0;
        Self {
            document,
            display_image: egui::ColorImage::from_rgba_unmultiplied([1, 1], &[0, 0, 0, 0]),
            display_texture: None,
            current_frame_index: 0,
            displayed_frame_index: 0,
            displayed_frame_ready: false,
            last_rendered_frame_index: None,
            displayed_diagnostics: None,
            display_size: source_size,
            shared,
            controller,
            selected_quad: false,
            show_quad_gizmo: false,
            was_playing: false,
            selected_asset: None,
            image_sizing: crate::ui::ImageSizing::FitCanvas,
            fitted_render_size: source_size,
            show_video_source_dialog: false,
            video_url: String::new(),
            pending_seek_target: None,
            last_pick_result: None,
            details_open: false,
        }
    }

    fn ensure_texture(&mut self, context: &egui::Context) {
        if self.display_texture.is_none() {
            self.display_texture = Some(context.load_texture(
                "video-editing-frame",
                self.display_image.clone(),
                egui::TextureOptions::LINEAR,
            ));
        }
    }

    fn video_source_dialog(&mut self, context: &egui::Context) {
        if !self.show_video_source_dialog {
            return;
        }
        let mut open = true;
        let mut close = false;
        egui::Window::new("Open video")
            .collapsible(false)
            .resizable(false)
            .open(&mut open)
            .show(context, |ui| {
                ui.set_min_width(420.0);
                ui.label("Select the video matched by this editing document.");
                if ui.button("Select local file...").clicked() {
                    self.shared.command.set(COMMAND_PICK_VIDEO);
                    close = true;
                }
                ui.separator();
                ui.label("Video URL");
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.video_url)
                        .hint_text("https://example.com/video.mp4")
                        .desired_width(f32::INFINITY),
                );
                let submit =
                    response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter));
                if ui.button("Load URL").clicked() || submit {
                    let url = self.video_url.trim();
                    if url.starts_with("https://") || url.starts_with("http://") {
                        self.shared.video_url_request.replace(Some(url.to_owned()));
                        close = true;
                    } else {
                        self.shared.error.replace(Some(
                            "video URL must start with http:// or https://".to_owned(),
                        ));
                    }
                }
                ui.weak("The URL must allow cross-origin video frame access.");
            });
        self.show_video_source_dialog = open && !close;
    }

    fn set_display_frame(&mut self, rendered: RenderedVideoFrame) {
        let frame = &rendered.frame;
        self.display_size = (frame.width, frame.height);
        self.displayed_frame_index = frame.frame_index;
        self.displayed_frame_ready = true;
        self.last_rendered_frame_index = Some(frame.frame_index);
        self.displayed_diagnostics = Some(rendered.diagnostics);
        if self.pending_seek_target == Some(frame.frame_index) {
            self.pending_seek_target = None;
        }
        self.display_image = egui::ColorImage::from_rgba_unmultiplied(
            [self.display_size.0 as usize, self.display_size.1 as usize],
            &frame.rgba,
        );
        if let Some(texture) = self.display_texture.as_mut() {
            texture.set(self.display_image.clone(), egui::TextureOptions::LINEAR);
        }
    }

    fn consume_video_frame(&mut self) {
        let frame = self.shared.frame.borrow_mut().take();
        let Some(frame) = frame else {
            return;
        };
        self.current_frame_index = frame.frame_index;
        self.shared.latest_video_frame.replace(Some(frame));
        self.shared.request_overlay();
        self.schedule_overlay();
    }

    fn consume_rendered_frame(&mut self) {
        let rendered = self.shared.rendered_frame.borrow_mut().take();
        let Some(rendered) = rendered else {
            return;
        };
        if self.shared.accepts_render(&rendered) {
            self.set_display_frame(rendered);
        }
    }

    fn consume_asset_defaults(&mut self) {
        let Some((asset, mode, material)) = self.shared.asset_defaults.borrow_mut().take() else {
            return;
        };
        if self.selected_asset == Some(asset) {
            self.controller.state.modes[0] = mode;
            self.controller.state.materials[0] = material;
            self.controller.state.environment_available = true;
            self.controller.state.lighting = match asset {
                CatalogAsset::Dragon => trd_core::Lighting {
                    ambient: 0.0,
                    scale: 0.0,
                },
                CatalogAsset::CocaColaCan | CatalogAsset::BeerCan => trd_core::Lighting::default(),
            };
            self.controller.rebase_reset();
            self.shared.request_overlay();
        }
    }

    fn consume_pick_result(&mut self) {
        let Some(result) = self.shared.pick_result.borrow_mut().take() else {
            return;
        };
        if !self.shared.accepts_pick(&result) {
            return;
        }
        let hit = result.hit;
        self.last_pick_result = Some(hit);
        if hit != self.controller.state.selected {
            self.controller.state.selected = hit;
            self.shared.request_overlay();
        }
    }

    fn schedule_overlay(&self) {
        if !self.shared.needs_overlay.get()
            || self.shared.render_in_flight.get()
            || self.shared.pick_in_flight.get()
        {
            return;
        }

        let Some(video) = self.shared.latest_video_frame.borrow().clone() else {
            return;
        };
        let Some(background_frame) = self
            .document
            .frames
            .get(video.frame_index as usize)
            .cloned()
        else {
            return;
        };
        let quad_frame = self.quad_frame_at(video.frame_index);
        let show_quad = !self.shared.video_playing.get() && background_frame.tracked;
        let quad_model = quad_frame
            .filter(|_| show_quad)
            .map(trd_placement::quad_outline_model);
        let quad_axes = quad_frame
            .filter(|_| show_quad)
            .map(trd_placement::quad_axes_model);
        let show_object =
            self.selected_asset.is_some() && self.selected_quad && background_frame.tracked;
        let placement_frame = show_object.then_some(background_frame.clone());
        let model = if show_object {
            self.placement_model_at(video.frame_index)
        } else {
            None
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            return;
        };
        let source_size = (self.document.video.width, self.document.video.height);
        let requested_size = match self.image_sizing {
            crate::ui::ImageSizing::FitCanvas => (
                self.fitted_render_size.0.min(source_size.0).max(1),
                self.fitted_render_size.1.min(source_size.1).max(1),
            ),
            crate::ui::ImageSizing::OriginalResolution => source_size,
        };
        if let Err(error) = renderer.resize(requested_size.0, requested_size.1) {
            self.shared.renderer.replace(Some(renderer));
            self.shared.error.replace(Some(error));
            return;
        }
        self.shared
            .renderer_diagnostics
            .replace(Some(renderer.diagnostics()));
        let render_size = renderer.size();
        self.shared.needs_overlay.set(false);
        let mut state = self.controller.state.clone();
        let rendered_playing = self.shared.video_playing.get();
        if rendered_playing {
            state.selected = None;
            state.show_aabb = false;
            state.show_axes = false;
            state.show_local_axes = false;
            state.show_world_grid = false;
            state.show_local_grid = false;
        }
        self.shared.render_in_flight.set(true);
        self.shared
            .render_in_flight_frame
            .set(Some(video.frame_index));
        let shared = self.shared.clone();
        let render_revision = shared.render_revision.get();
        let source_generation = video.source_generation;
        let renderer_generation = shared.renderer_generation.get();
        let width = self.document.video.width;
        let height = self.document.video.height;
        let show_quad_gizmo = self.show_quad_gizmo;
        let selected_asset = self.selected_asset;
        let selected_quad = self.selected_quad;
        let move_direction = self.controller.move_direction;
        let rendered_model = model;
        let background_frame_index = video.frame_index;
        let background_media_time = video.media_time_seconds;
        let render_started = Instant::now();
        let render = async move {
            let result = renderer
                .render(
                    &video.rgba,
                    video.width,
                    video.height,
                    (width, height),
                    &background_frame,
                    quad_model,
                    quad_axes,
                    show_quad_gizmo,
                    placement_frame.as_ref(),
                    model,
                    &state,
                )
                .await;
            let renderer_diagnostics = renderer.diagnostics();
            if shared.renderer_generation.get() != renderer_generation {
                shared.render_in_flight.set(false);
                shared.render_in_flight_frame.set(None);
                shared.request_repaint();
                return;
            }
            shared.renderer.replace(Some(renderer));
            let current = source_generation == shared.source_generation.get()
                && render_revision == shared.render_revision.get();
            match (current, result) {
                (true, Ok(rgba)) => {
                    shared.record_render_latency(render_started);
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics.clone()));
                    shared.last_render_error.replace(None);
                    shared.rendered_frame.replace(Some(RenderedVideoFrame {
                        frame: IncomingVideoFrame {
                            rgba,
                            width: render_size.0,
                            height: render_size.1,
                            frame_index: background_frame_index,
                            media_time_seconds: background_media_time,
                            source_generation,
                        },
                        render_revision,
                        diagnostics: RenderedFrameDiagnostics {
                            media_time_seconds: background_media_time,
                            scene: state,
                            selected_asset,
                            selected_quad,
                            move_direction,
                            playing: rendered_playing,
                            show_quad,
                            show_quad_gizmo,
                            draw_model: rendered_model,
                            renderer: renderer_diagnostics,
                        },
                    }));
                }
                (true, Err(error)) => {
                    shared.record_render_latency(render_started);
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_render_error.replace(Some(error.clone()));
                    shared.error.replace(Some(error));
                }
                (false, _) => {}
            }
            shared.render_in_flight.set(false);
            shared.render_in_flight_frame.set(None);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(render);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(render);
    }

    fn schedule_pick(&self) {
        if self.shared.render_in_flight.get() || self.shared.pick_in_flight.get() {
            return;
        }
        let Some(request) = self.shared.pending_pick.take() else {
            return;
        };
        let Some(frame) = self
            .document
            .frames
            .get(self.displayed_frame_index as usize)
            .cloned()
        else {
            return;
        };
        let Some(model) = self.placement_model_at(self.displayed_frame_index) else {
            return;
        };
        let Some(mut renderer) = self.shared.renderer.borrow_mut().take() else {
            self.shared.pending_pick.set(Some(request));
            return;
        };
        let source_size = (self.document.video.width, self.document.video.height);
        let render_size = renderer.size();
        let target_point = (
            request.point.0 * render_size.0 / self.display_size.0.max(1),
            request.point.1 * render_size.1 / self.display_size.1.max(1),
        );
        self.shared.pick_in_flight.set(true);
        let shared = self.shared.clone();
        let renderer_generation = shared.renderer_generation.get();
        let pick = async move {
            let result = renderer
                .pick(&frame, source_size, model, target_point)
                .await;
            let renderer_diagnostics = renderer.diagnostics();
            if shared.renderer_generation.get() != renderer_generation {
                shared.pick_in_flight.set(false);
                shared.request_repaint();
                return;
            }
            shared.renderer.replace(Some(renderer));
            let current = request.id == shared.pick_revision.get()
                && request.source_generation == shared.source_generation.get()
                && request.render_revision == shared.render_revision.get();
            match (current, result) {
                (true, Ok(hit)) => {
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_pick_error.replace(None);
                    shared.pick_result.replace(Some(PickResult {
                        id: request.id,
                        source_generation: request.source_generation,
                        render_revision: request.render_revision,
                        hit,
                    }));
                }
                (true, Err(error)) => {
                    shared
                        .renderer_diagnostics
                        .replace(Some(renderer_diagnostics));
                    shared.last_pick_error.replace(Some(error.clone()));
                    shared.error.replace(Some(error));
                }
                (false, _) => {}
            }
            shared.pick_in_flight.set(false);
            if let Some(context) = shared.context.borrow().as_ref() {
                context.request_repaint();
            }
        };
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_futures::spawn_local(pick);
        #[cfg(not(target_arch = "wasm32"))]
        pollster::block_on(pick);
    }

    fn quad_frame_at(&self, frame_index: u32) -> Option<trd_placement::QuadFrame> {
        self.quad_frame_result_at(frame_index).ok()
    }

    fn quad_frame_result_at(
        &self,
        frame_index: u32,
    ) -> Result<trd_placement::QuadFrame, TrackingPlacementError> {
        let frame = self
            .document
            .frames
            .get(frame_index as usize)
            .ok_or(TrackingPlacementError::FrameOutOfRange)?;
        let k = frame.k.ok_or(TrackingPlacementError::MissingIntrinsics)?;
        let placement_quad = frame
            .placement_quad
            .ok_or(TrackingPlacementError::MissingQuad)?;
        trd_placement::quad_frame(
            trd_placement::CameraIntrinsics { row_major: k },
            trd_placement::PlacementQuad {
                points_px: placement_quad,
            },
        )
        .map_err(TrackingPlacementError::from)
    }

    fn placement_model_at(&self, frame_index: u32) -> Option<trd_core::Matrix4> {
        let frame = self.quad_frame_at(frame_index)?;
        let placement = trd_placement::LocalPlacement {
            offset_e1: 1.3,
            offset_e2: -1.7,
            size_factor: 0.24,
            ..Default::default()
        };
        let quad_basis = trd_placement::placement_model(frame, placement).ok()?;
        let object = self.controller.state.objects.first()?;
        let object_model = trd_core::Matrix4::from_cols_array(&object.model_matrix());
        Some(quad_basis * object_model)
    }

    fn diagnostics(&self) -> VideoEditingDiagnostics {
        let video = &self.document.video;
        let source = self.shared.video_source.borrow().clone();
        let metadata = self.shared.video_metadata.get();
        let media = self.shared.video_media.get();
        let renderer = self
            .displayed_diagnostics
            .as_ref()
            .map(|displayed| displayed.renderer.clone())
            .or_else(|| self.shared.renderer_diagnostics.borrow().clone());
        let presented_frame_index = self
            .shared
            .latest_video_frame
            .borrow()
            .as_ref()
            .map(|frame| frame.frame_index);
        let displayed_frame_index = self
            .displayed_frame_ready
            .then_some(self.displayed_frame_index);
        let timeline_frame =
            displayed_frame_index.and_then(|index| self.document.frames.get(index as usize));
        // Media time rides with its own frame, so the timeline block describes
        // the frame actually on screen rather than a newer presented one.
        let displayed_media_time = self
            .displayed_frame_ready
            .then(|| {
                self.displayed_diagnostics
                    .as_ref()
                    .map(|displayed| displayed.media_time_seconds)
            })
            .flatten();
        let render_count = self.shared.render_latency_count.get();
        let in_flight_frame_index = self.shared.render_in_flight_frame.get();
        let coalesced_frame_index = in_flight_frame_index.and_then(|in_flight| {
            presented_frame_index.filter(|presented| *presented != in_flight)
        });

        let (quad_frame, placement_error) = match displayed_frame_index {
            Some(index) if timeline_frame.is_some_and(|frame| frame.tracked) => {
                match self.quad_frame_result_at(index) {
                    Ok(frame) => (Some(frame), None),
                    Err(error) => (None, Some(error)),
                }
            }
            _ => (None, None),
        };
        let previous_quad = displayed_frame_index.and_then(|index| {
            (0..index).rev().find_map(|previous_index| {
                self.document
                    .frames
                    .get(previous_index as usize)
                    .filter(|frame| frame.tracked)
                    .and_then(|_| {
                        self.quad_frame_at(previous_index)
                            .map(|frame| (previous_index, frame))
                    })
            })
        });
        let pose_delta =
            quad_frame
                .zip(previous_quad)
                .map(|(current, (previous_frame_index, previous))| {
                    pose_delta(previous_frame_index, previous, current)
                });
        let normal_sign_warning = quad_frame
            .zip(previous_quad)
            .is_some_and(|(current, (_, previous))| dot3(current.e3, previous.e3) < 0.0);
        let quad_frame_diagnostics = quad_frame.map(quad_frame_diagnostics);

        let scene = self
            .displayed_diagnostics
            .as_ref()
            .map_or(&self.controller.state, |displayed| &displayed.scene);
        let selected_asset = self
            .displayed_diagnostics
            .as_ref()
            .map_or(self.selected_asset, |displayed| displayed.selected_asset);
        let selected_quad = self
            .displayed_diagnostics
            .as_ref()
            .map_or(self.selected_quad, |displayed| displayed.selected_quad);
        let playing = self
            .displayed_diagnostics
            .as_ref()
            .is_some_and(|displayed| displayed.playing);
        let object = scene.objects[0];
        let asset = renderer.as_ref().and_then(|facts| facts.asset.as_ref());
        let material = &scene.materials[0];
        let imported_material = asset.map(|facts| &facts.imported_material);
        let ibl = scene.image_based_lighting[0];
        let tone_mapping = scene.tone_mappings[0];
        let pbr_debug_view = scene.pbr_debug_views[0];
        let render_mode = scene.modes[0];
        let tracked = timeline_frame.is_some_and(|frame| frame.tracked);
        let visibility_reason = if playing {
            "playing"
        } else if selected_asset.is_none() {
            "no asset"
        } else if !tracked {
            "untracked tail"
        } else if !selected_quad {
            "no quad selected"
        } else {
            "tracked"
        };
        let object_visible = visibility_reason == "tracked";
        let draw_model = self
            .displayed_diagnostics
            .as_ref()
            .and_then(|displayed| displayed.draw_model)
            .or_else(|| {
                displayed_frame_index
                    .filter(|_| object_visible)
                    .and_then(|index| self.placement_model_at(index))
            })
            .map(trd_core::Matrix4::to_cols_array);
        let move_direction = self
            .displayed_diagnostics
            .as_ref()
            .map_or(self.controller.move_direction, |displayed| {
                displayed.move_direction
            });
        let movement_basis = match move_direction {
            crate::interaction::MoveDirection::LocalX
            | crate::interaction::MoveDirection::LocalY
            | crate::interaction::MoveDirection::LocalZ => ["object X", "object Y", "object Z"],
            _ => ["quad e1", "quad e2", "quad e3"],
        };
        let reflective_tracking_warning = imported_material
            .is_some_and(|imported| imported.metallic >= 0.7 || imported.auxiliary.textures.normal)
            && pose_delta.as_ref().is_some_and(|delta| {
                delta.rotation_degrees >= 1.0
                    || quad_frame.is_some_and(|quad| delta.translation >= quad.axis_length * 0.02)
            });

        let show_quad = self
            .displayed_diagnostics
            .as_ref()
            .is_some_and(|displayed| displayed.show_quad);
        let show_quad_gizmo = self
            .displayed_diagnostics
            .as_ref()
            .is_some_and(|displayed| displayed.show_quad_gizmo);
        let background_drawables =
            1 + u32::from(show_quad) + if show_quad && show_quad_gizmo { 2 } else { 0 };
        let foreground_drawables = if object_visible {
            1 + u32::from(scene.show_local_axes)
                + u32::from(scene.show_axes)
                + u32::from(scene.show_local_grid)
                + u32::from(scene.show_world_grid)
        } else {
            0
        };
        let selection_drawables =
            if object_visible && (scene.show_aabb || scene.selected == Some(0)) {
                1
            } else {
                0
            };
        let render_target_size = renderer
            .as_ref()
            .map(|facts| facts.target_size)
            .unwrap_or(self.display_size);
        let upload_bytes = self
            .shared
            .latest_video_frame
            .borrow()
            .as_ref()
            .map(|frame| frame.rgba.len() as u64);

        VideoEditingDiagnostics {
            source: SourceDiagnostics {
                expected_name: video.source_name.clone(),
                expected_byte_length: video.byte_length,
                expected_mime: video.mime.clone(),
                expected_codec: video.codec.clone(),
                expected_size: [video.width, video.height],
                expected_fps: [video.fps_num, video.fps_den],
                expected_frame_count: video.frame_count,
                expected_duration_seconds: video.duration_us as f64 / 1_000_000.0,
                expected_sha256: video.sha256.clone(),
                observed_kind: source.as_ref().map(|source| source.kind),
                observed_name: source.as_ref().map(|source| source.name.clone()),
                observed_byte_length: source.as_ref().and_then(|source| source.byte_length),
                observed_size: metadata.map(|metadata| [metadata.width, metadata.height]),
                observed_duration_seconds: metadata.map(|metadata| metadata.duration_seconds),
                ready_state: media.ready_state,
                loaded: self.shared.video_loaded.get(),
                playing: self.shared.video_playing.get(),
                ended: media.ended,
                error: self.shared.error.borrow().clone(),
                digest_status: "not browser-verified yet",
            },
            timeline: TimelineDiagnostics {
                media_time_seconds: displayed_media_time,
                requested_frame_index: self.current_frame_index,
                presented_frame_index,
                displayed_frame_index,
                rendered_frame_index: self.last_rendered_frame_index,
                arrow_video_frame_index: timeline_frame.map(|frame| frame.video_frame_index),
                present_index: timeline_frame.map(|frame| frame.present_index),
                timestamp_us: timeline_frame.map(|frame| frame.timestamp_us),
                media_timestamp_delta_ms: displayed_media_time.zip(timeline_frame).map(
                    |(media_time, frame)| {
                        (media_time - frame.timestamp_us as f64 / 1_000_000.0) * 1_000.0
                    },
                ),
                tracked: timeline_frame.map(|frame| frame.tracked),
                source_size: [video.width, video.height],
                render_size: [render_target_size.0, render_target_size.1],
                source_generation: self.shared.source_generation.get(),
                render_revision: self.shared.render_revision.get(),
                pending_render_generation: self
                    .shared
                    .needs_overlay
                    .get()
                    .then_some(self.shared.render_revision.get()),
                in_flight_frame_index,
                coalesced_frame_index,
                last_render_latency_ms: self.shared.last_render_latency_ms.get(),
                average_render_latency_ms: (render_count > 0)
                    .then(|| self.shared.render_latency_total_ms.get() / render_count as f64),
                seek_target: self.pending_seek_target,
                seek_pending: self.pending_seek_target.is_some(),
            },
            tracking: TrackingDiagnostics {
                points_tl_tr_br_bl: timeline_frame.and_then(|frame| frame.placement_quad),
                intrinsics_fx_fy_cx_cy: timeline_frame
                    .and_then(|frame| frame.k)
                    .map(|k| [k[0], k[4], k[2], k[5]]),
                quad_frame: quad_frame_diagnostics,
                pose_delta,
                normal_sign_warning,
                placement_error,
                smoothing: "off",
            },
            placement: PlacementDiagnostics {
                selected_quad,
                selected_object: scene.selected,
                catalog_asset: selected_asset.map(CatalogAsset::label),
                source_format: asset.map(|facts| facts.source_format),
                preview_aabb_min: asset.map(|facts| facts.aabb_min),
                preview_aabb_max: asset.map(|facts| facts.aabb_max),
                preview_scale: asset.map(|facts| facts.preview_scale),
                preset_size_factor: 0.24,
                preset_offset_e1: 1.3,
                preset_offset_e2: -1.7,
                preset_lift: 1.0,
                object_translation: object.translation,
                object_rotation_degrees: [
                    object.yaw.to_degrees(),
                    object.pitch.to_degrees(),
                    object.roll.to_degrees(),
                ],
                object_scale: object.scale,
                movement_basis,
                draw_model,
                visibility_reason,
            },
            material_lighting: MaterialLightingDiagnostics {
                render_mode: render_mode_label(render_mode),
                imported_metallic: imported_material.map(|material| material.metallic),
                imported_roughness: imported_material.map(|material| material.roughness),
                base_color_map: imported_material
                    .is_some_and(|material| material.auxiliary.textures.base_color),
                metallic_roughness_map: imported_material
                    .is_some_and(|material| material.auxiliary.textures.metallic_roughness),
                normal_map: imported_material
                    .is_some_and(|material| material.auxiliary.textures.normal),
                metallic: material.metallic,
                roughness: material.roughness,
                specular: material.specular,
                clearcoat: material.clearcoat,
                environment_name: scene.environment_available.then_some("uffizi-large.hdr"),
                environment_intensity: ibl.intensity,
                environment_rotation_degrees: ibl.rotation.to_degrees(),
                direct_light_scale: scene.lighting.scale,
                ambient: scene.lighting.ambient,
                exposure: tone_mapping.exposure,
                tone_map: tone_map_label(tone_mapping.operator),
                pbr_debug_view: pbr_debug_view_label(pbr_debug_view),
                tracking_warning: reflective_tracking_warning
                    .then_some("reflective/normal-mapped material may amplify raw tracking jitter"),
            },
            renderer: RendererDiagnostics {
                adapter_name: renderer.as_ref().map(|facts| facts.adapter_name.clone()),
                backend: renderer.as_ref().map(|facts| facts.backend.clone()),
                device_type: renderer.as_ref().map(|facts| facts.device_type.clone()),
                source_size: [video.width, video.height],
                render_target_size: [render_target_size.0, render_target_size.1],
                mode: render_mode_label(render_mode),
                msaa_samples: renderer.as_ref().map(|facts| facts.msaa_samples),
                background_drawables,
                foreground_drawables,
                selection_drawables,
                frame_texture_upload_bytes: upload_bytes,
                pick_target_size: renderer
                    .as_ref()
                    .and_then(|facts| facts.pick_target_size)
                    .map(|(width, height)| [width, height]),
                latest_pick_result: self.last_pick_result,
                last_render_error: self.shared.last_render_error.borrow().clone(),
                last_pick_error: self.shared.last_pick_error.borrow().clone(),
            },
        }
    }
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length3(value: [f32; 3]) -> f32 {
    dot3(value, value).sqrt()
}

fn quad_frame_diagnostics(frame: trd_placement::QuadFrame) -> QuadFrameDiagnostics {
    QuadFrameDiagnostics {
        origin: frame.origin_camera,
        e1: frame.e1,
        e2: frame.e2,
        e3: frame.e3,
        half_edge_lengths: [length3(frame.half_edge1), length3(frame.half_edge2)],
        axis_length: frame.axis_length,
        orthogonality_errors: [
            dot3(frame.e1, frame.e2).abs(),
            dot3(frame.e1, frame.e3).abs(),
            dot3(frame.e2, frame.e3).abs(),
        ],
        handedness_determinant: dot3(frame.e1, cross3(frame.e2, frame.e3)),
    }
}

fn pose_delta(
    previous_frame_index: u32,
    previous: trd_placement::QuadFrame,
    current: trd_placement::QuadFrame,
) -> PoseDeltaDiagnostics {
    let translation = length3([
        current.origin_camera[0] - previous.origin_camera[0],
        current.origin_camera[1] - previous.origin_camera[1],
        current.origin_camera[2] - previous.origin_camera[2],
    ]);
    let trace = dot3(previous.e1, current.e1)
        + dot3(previous.e2, current.e2)
        + dot3(previous.e3, current.e3);
    let rotation_degrees = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0).acos().to_degrees();
    PoseDeltaDiagnostics {
        previous_frame_index,
        translation,
        rotation_degrees,
        axis_length_ratio: current.axis_length / previous.axis_length.max(f32::EPSILON),
    }
}

fn render_mode_label(mode: trd_core::RenderMode) -> &'static str {
    match mode {
        trd_core::RenderMode::Filled => "filled",
        trd_core::RenderMode::Wireframe => "wireframe",
        trd_core::RenderMode::Textured => "textured",
        trd_core::RenderMode::Pbr => "pbr",
        trd_core::RenderMode::Shadow => "shadow",
    }
}

fn tone_map_label(operator: trd_core::Tonemap) -> &'static str {
    match operator {
        trd_core::Tonemap::Reinhard => "reinhard",
        trd_core::Tonemap::Aces => "aces",
    }
}

fn pbr_debug_view_label(view: trd_core::PbrDebugView) -> &'static str {
    match view {
        trd_core::PbrDebugView::Shaded => "shaded",
        trd_core::PbrDebugView::Roughness => "roughness",
        trd_core::PbrDebugView::Metallic => "metallic",
        trd_core::PbrDebugView::Normal => "normal",
    }
}

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
        let diagnostics = self.details_open.then(|| self.diagnostics());
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
            let response = ui.collapsing("Details", |ui| {
                if let Some(diagnostics) = diagnostics.as_ref() {
                    video_editing_diagnostics_ui(ui, diagnostics);
                } else {
                    ui.weak("Loading diagnostics...");
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

/// Maps a media-clock time to the nearest zero-based video frame, clamped to
/// the editing document's declared frame range.
pub fn frame_index_at_media_time(
    media_time_seconds: f64,
    fps_num: u32,
    fps_den: u32,
    frame_count: u32,
) -> u32 {
    let frame = (media_time_seconds * f64::from(fps_num) / f64::from(fps_den.max(1)))
        .round()
        .max(0.0) as u32;
    frame.min(frame_count.saturating_sub(1))
}

/// Maps a zero-based video frame to its media-clock time, clamped to the editing
/// document's declared frame range.
pub fn media_time_at_frame(frame_index: u32, fps_num: u32, fps_den: u32, frame_count: u32) -> f64 {
    let frame = frame_index.min(frame_count.saturating_sub(1));
    f64::from(frame) * f64::from(fps_den) / f64::from(fps_num.max(1))
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

fn point_in_quad(point: [f32; 2], quad: [[f32; 2]; 4]) -> bool {
    let mut inside = false;
    let mut previous = quad[3];
    for current in quad {
        if (current[1] > point[1]) != (previous[1] > point[1])
            && point[0]
                < (previous[0] - current[0]) * (point[1] - current[1]) / (previous[1] - current[1])
                    + current[0]
        {
            inside = !inside;
        }

        previous = current;
    }
    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> trd_core::VideoEditingDocument {
        trd_core::VideoEditingDocument {
            video: trd_core::VideoInfo {
                source_name: "shot.mp4".to_owned(),
                mime: "video/mp4".to_owned(),
                codec: "h264".to_owned(),
                sha256: "unused".to_owned(),
                byte_length: 1,
                width: 1920,
                height: 1080,
                fps_num: 24,
                fps_den: 1,
                frame_count: 288,
                duration_us: 12_000_000,
            },
            poster_bytes: vec![1, 2, 3],
            frames: vec![trd_core::VideoEditingFrame {
                video_frame_index: 0,
                present_index: 0,
                timestamp_us: 0,
                k: None,
                placement_quad: None,
                tracked: false,
            }],
        }
    }

    #[test]
    fn unloaded_editor_starts_without_a_frame_or_texture() {
        let shared = Rc::new(VideoEditingShared::default());
        let app = VideoEditingApp::new(document(), shared.clone());
        assert!(shared.latest_video_frame.borrow().is_none());
        assert!(app.display_texture.is_none());
        assert_eq!(app.display_size, (1920, 1080));
    }

    #[test]
    fn unloaded_player_status_is_zeroed() {
        let document = document();
        assert_eq!(
            player_status_label(false, 42, &document.video),
            "00:00 / 00:00  ·  frame 0/0"
        );
    }

    #[test]
    fn newest_incoming_frame_replaces_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        shared
            .update_video_frame_rgba(vec![5, 6, 7, 8], 1, 1, 9, 0.5)
            .unwrap();

        let frame = shared.frame.borrow_mut().take().unwrap();
        assert_eq!(frame.frame_index, 9);
        assert_eq!(frame.rgba, vec![5, 6, 7, 8]);
    }

    #[test]
    fn invalid_incoming_frame_does_not_replace_the_pending_frame() {
        let shared = VideoEditingShared::default();
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        assert!(shared
            .update_video_frame_rgba(vec![5, 6, 7], 1, 1, 9, 0.5)
            .is_err());

        assert_eq!(
            shared
                .frame
                .borrow()
                .as_ref()
                .map(|frame| frame.frame_index),
            Some(7)
        );
    }

    #[test]
    fn one_slot_commands_and_seek_requests_keep_the_newest_value() {
        let shared = VideoEditingShared::default();
        shared.command.set(COMMAND_PLAY);
        shared.command.set(COMMAND_PAUSE);
        assert_eq!(shared.take_command(), Some(VideoEditingCommand::Pause));
        assert_eq!(shared.take_command(), None);

        shared.seek_frame.set(12);
        shared.seek_frame.set(42);
        assert_eq!(shared.take_seek_frame(), Some(42));
        assert_eq!(shared.take_seek_frame(), None);
    }

    #[test]
    fn media_time_frame_mapping_rounds_and_clamps_at_boundaries() {
        assert_eq!(frame_index_at_media_time(-1.0, 24, 1, 288), 0);
        assert_eq!(frame_index_at_media_time(1.0 / 48.0, 24, 1, 288), 1);
        assert_eq!(frame_index_at_media_time(30.0, 24, 1, 288), 287);
        assert_eq!(media_time_at_frame(288, 24, 1, 288), 287.0 / 24.0);
    }

    #[test]
    fn source_reset_invalidates_frames_renders_and_picks() {
        let shared = VideoEditingShared::default();
        shared.set_video_status(true, false);
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 7, 0.25)
            .unwrap();
        let source_generation = shared.source_generation.get();
        shared.request_overlay();
        let render_revision = shared.render_revision.get();
        shared.request_pick((3, 4));
        let pick = shared.pending_pick.get().unwrap();
        let rendered = RenderedVideoFrame {
            frame: IncomingVideoFrame {
                rgba: vec![1, 2, 3, 4],
                width: 1,
                height: 1,
                frame_index: 7,
                media_time_seconds: 0.25,
                source_generation,
            },
            render_revision,
            diagnostics: test_rendered_frame_diagnostics(),
        };
        let pick_result = PickResult {
            id: pick.id,
            source_generation,
            render_revision,
            hit: Some(0),
        };
        assert!(shared.accepts_render(&rendered));
        assert!(shared.accepts_pick(&pick_result));

        shared.set_video_status(false, false);
        assert!(!shared.accepts_render(&rendered));
        assert!(!shared.accepts_pick(&pick_result));
        assert!(shared.frame.borrow().is_none());
        assert!(shared.latest_video_frame.borrow().is_none());
        assert!(shared.pending_pick.get().is_none());
    }

    #[test]
    fn newer_scene_revision_invalidates_render_and_pick_completions() {
        let shared = VideoEditingShared::default();
        shared.request_overlay();
        let revision = shared.render_revision.get();
        shared.request_pick((3, 4));
        let pick = shared.pending_pick.get().unwrap();
        let rendered = RenderedVideoFrame {
            frame: IncomingVideoFrame {
                rgba: vec![1, 2, 3, 4],
                width: 1,
                height: 1,
                frame_index: 7,
                media_time_seconds: 0.25,
                source_generation: shared.source_generation.get(),
            },
            render_revision: revision,
            diagnostics: test_rendered_frame_diagnostics(),
        };
        let pick_result = PickResult {
            id: pick.id,
            source_generation: shared.source_generation.get(),
            render_revision: revision,
            hit: Some(0),
        };
        shared.request_overlay();
        assert!(!shared.accepts_render(&rendered));
        assert!(!shared.accepts_pick(&pick_result));
    }

    #[test]
    fn newer_pick_request_invalidates_older_pick_completion() {
        let shared = VideoEditingShared::default();
        shared.request_pick((1, 2));
        let first = shared.pending_pick.get().unwrap();
        shared.request_pick((3, 4));
        let result = PickResult {
            id: first.id,
            source_generation: first.source_generation,
            render_revision: first.render_revision,
            hit: Some(0),
        };
        assert!(!shared.accepts_pick(&result));
        assert_eq!(shared.pending_pick.get().unwrap().point, (3, 4));
    }

    #[test]
    fn diagnostics_keep_the_scene_bound_to_the_displayed_render() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared);
        let mut rendered = test_rendered_frame_diagnostics();
        rendered.selected_asset = Some(CatalogAsset::Dragon);
        rendered.scene.materials[0].metallic = 0.25;
        rendered.scene.lighting = trd_core::Lighting {
            ambient: 0.0,
            scale: 0.0,
        };
        rendered.scene.environment_available = true;
        rendered.renderer.asset = Some(crate::video_editing_renderer::ImportedAssetDiagnostics {
            source_format: "GLB",
            aabb_min: [-1.0; 3],
            aabb_max: [1.0; 3],
            preview_scale: 1.0,
            imported_material: trd_core::DisneyMaterial {
                metallic: 1.0,
                auxiliary: trd_core::Auxiliary {
                    textures: trd_core::MaterialTextures {
                        metallic_roughness: true,
                        normal: true,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        });
        app.displayed_frame_ready = true;
        app.displayed_frame_index = 0;
        app.last_rendered_frame_index = Some(0);
        app.displayed_diagnostics = Some(rendered);
        app.controller.state.materials[0].metallic = 0.9;

        let diagnostics = app.diagnostics();
        assert_eq!(diagnostics.timeline.displayed_frame_index, Some(0));
        assert_eq!(diagnostics.material_lighting.metallic, 0.25);
        assert_eq!(diagnostics.material_lighting.imported_metallic, Some(1.0));
        assert!(diagnostics.material_lighting.metallic_roughness_map);
        assert!(diagnostics.material_lighting.normal_map);
        assert_eq!(diagnostics.material_lighting.direct_light_scale, 0.0);
        assert_eq!(
            diagnostics.material_lighting.environment_name,
            Some("uffizi-large.hdr")
        );
    }

    #[test]
    fn diagnostics_media_time_tracks_the_displayed_frame_not_a_newer_one() {
        let shared = Rc::new(VideoEditingShared::default());
        let mut app = VideoEditingApp::new(document(), shared.clone());
        assert_eq!(app.diagnostics().timeline.media_time_seconds, None);

        let mut rendered = test_rendered_frame_diagnostics();
        rendered.media_time_seconds = 0.0;
        app.displayed_frame_ready = true;
        app.displayed_frame_index = 0;
        app.last_rendered_frame_index = Some(0);
        app.displayed_diagnostics = Some(rendered);

        // A newer frame arrives but has not reached the screen: the timeline
        // block must still describe frame 0, delta included.
        shared
            .update_video_frame_rgba(vec![1, 2, 3, 4], 1, 1, 5, 5.0 / 24.0)
            .unwrap();
        shared.set_video_media_observation(4, false);

        let diagnostics = app.diagnostics();
        assert_eq!(diagnostics.timeline.presented_frame_index, None);
        assert_eq!(diagnostics.timeline.displayed_frame_index, Some(0));
        assert_eq!(diagnostics.timeline.media_time_seconds, Some(0.0));
        assert_eq!(diagnostics.timeline.media_timestamp_delta_ms, Some(0.0));
        assert_eq!(diagnostics.source.ready_state, 4);
    }

    #[test]
    fn pose_delta_reports_unsmoothed_translation_rotation_and_scale() {
        let previous = trd_placement::QuadFrame {
            origin_camera: [0.0, 0.0, 1.0],
            e1: [1.0, 0.0, 0.0],
            e2: [0.0, 1.0, 0.0],
            e3: [0.0, 0.0, 1.0],
            half_edge1: [0.5, 0.0, 0.0],
            half_edge2: [0.0, 0.5, 0.0],
            axis_length: 0.5,
        };
        let current = trd_placement::QuadFrame {
            origin_camera: [0.0, 0.3, 1.4],
            e1: [0.0, 1.0, 0.0],
            e2: [-1.0, 0.0, 0.0],
            e3: [0.0, 0.0, 1.0],
            half_edge1: [1.0, 0.0, 0.0],
            half_edge2: [0.0, 1.0, 0.0],
            axis_length: 1.0,
        };

        let delta = pose_delta(6, previous, current);
        assert_eq!(delta.previous_frame_index, 6);
        assert!((delta.translation - 0.5).abs() < 1e-6);
        assert!((delta.rotation_degrees - 90.0).abs() < 1e-4);
        assert!((delta.axis_length_ratio - 2.0).abs() < 1e-6);
    }

    fn test_rendered_frame_diagnostics() -> RenderedFrameDiagnostics {
        RenderedFrameDiagnostics {
            media_time_seconds: 0.25,
            scene: crate::scene::SceneState::default(),
            selected_asset: None,
            selected_quad: false,
            move_direction: crate::interaction::MoveDirection::Reference1,
            playing: false,
            show_quad: false,
            show_quad_gizmo: false,
            draw_model: None,
            renderer: crate::video_editing_renderer::VideoRendererDiagnostics {
                adapter_name: "test".to_owned(),
                backend: "test".to_owned(),
                device_type: "test".to_owned(),
                target_size: (1, 1),
                pick_target_size: None,
                msaa_samples: 4,
                asset: None,
            },
        }
    }
}
