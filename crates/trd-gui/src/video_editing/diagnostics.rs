//! Pure domain calculations behind the Details inspector (#167/#175).
//!
//! There is no snapshot DTO: [`super::details_ui`] reads app state directly and
//! calls these helpers at draw time, so the domain maths stays out of the UI.

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

/// Renders a column-major 4x4 as four readable rows.
pub(super) fn format_matrix(matrix: [f32; 16]) -> String {
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
        trd_core::RenderMode::Shaded => "pbr",
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
