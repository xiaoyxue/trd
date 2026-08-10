//! Immutable diagnostic snapshot for the video editor (#167).
//!
//! Pure domain data and calculations only: these types describe the frame that
//! reached the screen and are built once per repaint by
//! [`VideoEditingApp::diagnostics`](super::VideoEditingApp). Rendering them is
//! [`super::diagnostics_ui`]'s job, so UI code never rederives domain math.

use super::VideoSourceKind;

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
    pub(super) fn to_json(&self) -> Result<String, serde_json::Error> {
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

pub(super) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
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

pub(super) fn quad_frame_diagnostics(frame: trd_placement::QuadFrame) -> QuadFrameDiagnostics {
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

pub(super) fn pose_delta(
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

pub(super) fn render_mode_label(mode: trd_core::RenderMode) -> &'static str {
    match mode {
        trd_core::RenderMode::Filled => "filled",
        trd_core::RenderMode::Wireframe => "wireframe",
        trd_core::RenderMode::Textured => "textured",
        trd_core::RenderMode::Pbr => "pbr",
        trd_core::RenderMode::Shadow => "shadow",
    }
}

pub(super) fn tone_map_label(operator: trd_core::Tonemap) -> &'static str {
    match operator {
        trd_core::Tonemap::Reinhard => "reinhard",
        trd_core::Tonemap::Aces => "aces",
    }
}

pub(super) fn pbr_debug_view_label(view: trd_core::PbrDebugView) -> &'static str {
    match view {
        trd_core::PbrDebugView::Shaded => "shaded",
        trd_core::PbrDebugView::Roughness => "roughness",
        trd_core::PbrDebugView::Metallic => "metallic",
        trd_core::PbrDebugView::Normal => "normal",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
