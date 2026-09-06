//! Quad-local placement authored from camera intrinsics and image-space quads.
//!
//! This is the Rust port target of `examples/placement_quad_by_local_coord.py`.
//! It is deliberately CPU-only: callers turn its resolved models into ordinary
//! [`trd_core::Draw`] values using their uploaded mesh identities.

use thiserror::Error;
use trd_core::Matrix4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraIntrinsics {
    /// Row-major OpenCV K.
    pub row_major: [f32; 9],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacementQuad {
    /// Consecutive image points in TL, TR, BR, BL order.
    pub points_px: [[f32; 2]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadFrame {
    pub origin_camera: [f32; 3],
    pub e1: [f32; 3],
    pub e2: [f32; 3],
    pub e3: [f32; 3],
    /// Half edge from quad center along e1-like frame direction.
    pub half_edge1: [f32; 3],
    /// Half edge from quad center along e2-like frame direction.
    pub half_edge2: [f32; 3],
    pub axis_length: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalPlacement {
    pub offset_e1: f32,
    pub offset_e2: f32,
    pub lift: f32,
    pub size_factor: f32,
    pub yaw: f32,
    pub local_translation: [f32; 3],
    pub local_scale: f32,
}

impl Default for LocalPlacement {
    fn default() -> Self {
        Self {
            offset_e1: 0.0,
            offset_e2: 0.0,
            lift: 1.0,
            size_factor: 1.0,
            yaw: 0.0,
            local_translation: [0.0; 3],
            local_scale: 1.0,
        }
    }
}

#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum PlacementError {
    #[error("quad is degenerate")]
    DegenerateQuad,
    #[error("camera intrinsics are singular")]
    SingularIntrinsics,
    #[error("quad resolves behind the camera")]
    BehindCamera,
    #[error("placement scale must be positive")]
    InvalidScale,
}

/// Reconstructs the camera-space orthonormal local frame used by the Python
/// reference. Its axis/sign convention matches `normal_basis_from_quad`.
pub fn quad_frame(
    intrinsics: CameraIntrinsics,
    quad: PlacementQuad,
) -> Result<QuadFrame, PlacementError> {
    let k = column_major_intrinsics(intrinsics.row_major);
    let k_inv = inverse3(k).ok_or(PlacementError::SingularIntrinsics)?;
    let h = homography_unit_square_to_quad(quad.points_px)?;
    let b = mat3_mul(k_inv, h);
    let b0 = column3(b, 0);
    let b1 = column3(b, 1);
    let b2 = column3(b, 2);
    let scale = 1.0 / length(b0).max(1e-12);
    let mut r1 = scale3(b0, scale);
    let mut r2 = scale3(b1, scale);
    let mut t = scale3(b2, scale);
    if t[2] < 0.0 {
        r1 = scale3(r1, -1.0);
        r2 = scale3(r2, -1.0);
        t = scale3(t, -1.0);
    }
    if t[2] <= 0.0 {
        return Err(PlacementError::BehindCamera);
    }

    let mut e3 = normalize(cross(r1, r2)).ok_or(PlacementError::DegenerateQuad)?;
    let e1 = normalize(sub(r1, scale3(e3, dot(r1, e3)))).ok_or(PlacementError::DegenerateQuad)?;
    let mut e2 = normalize(cross(e3, e1)).ok_or(PlacementError::DegenerateQuad)?;
    let origin = add(add(scale3(r1, 0.5), scale3(r2, 0.5)), t);
    let origin_px = project(k, origin).ok_or(PlacementError::BehindCamera)?;
    let axis_length = 0.5 * length(r1);
    if project(k, add(origin, scale3(e3, axis_length))).ok_or(PlacementError::BehindCamera)?[1]
        > origin_px[1]
    {
        e3 = scale3(e3, -1.0);
        e2 = normalize(cross(e3, e1)).ok_or(PlacementError::DegenerateQuad)?;
    }
    Ok(QuadFrame {
        origin_camera: origin,
        e1,
        e2,
        e3,
        half_edge1: scale3(r1, 0.5),
        half_edge2: scale3(r2, 0.5),
        axis_length,
    })
}

/// Matches the Python placement formula and returns a column-major GL camera
/// model (`C4 * m_cam`) ready for a 0.0.6 `draw_model` entry.
pub fn placement_model(frame: QuadFrame, edit: LocalPlacement) -> Result<Matrix4, PlacementError> {
    if edit.size_factor <= 0.0 || edit.local_scale <= 0.0 {
        return Err(PlacementError::InvalidScale);
    }

    let size = frame.axis_length * edit.size_factor * edit.local_scale;
    let anchor = add(
        add(
            frame.origin_camera,
            scale3(frame.half_edge1, edit.offset_e1),
        ),
        scale3(frame.half_edge2, edit.offset_e2),
    );
    let local_translation = add(
        add(
            scale3(frame.e1, edit.local_translation[0]),
            scale3(frame.e3, edit.local_translation[1]),
        ),
        scale3(scale3(frame.e2, -1.0), edit.local_translation[2]),
    );
    let translation = add(
        add(anchor, scale3(frame.e3, edit.lift * size)),
        scale3(local_translation, size),
    );
    // `[e1, e3, -e2] * rotate_y(yaw)`, exactly as the Python reference composes
    // it. The first column takes `+sin * e2` — writing `-e2` there instead
    // shears the basis (`x · z == sin 2·yaw`), collapses it at ±45° and mirrors
    // it beyond, because the result is then no longer a rotation.
    let (sin, cos) = edit.yaw.sin_cos();
    let x = add(scale3(frame.e1, cos), scale3(frame.e2, sin));
    let z = add(scale3(frame.e1, sin), scale3(scale3(frame.e2, -1.0), cos));
    let model_camera = [
        x[0] * size,
        x[1] * size,
        x[2] * size,
        0.0,
        frame.e3[0] * size,
        frame.e3[1] * size,
        frame.e3[2] * size,
        0.0,
        z[0] * size,
        z[1] * size,
        z[2] * size,
        0.0,
        translation[0],
        translation[1],
        translation[2],
        1.0,
    ];
    let cv_to_gl = [
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    Ok(Matrix4::from_cols_array(&cv_to_gl) * Matrix4::from_cols_array(&model_camera))
}

/// Projects a camera-space point using row-major OpenCV intrinsics.
pub fn project_camera(
    intrinsics: CameraIntrinsics,
    point_camera: [f32; 3],
) -> Result<[f32; 2], PlacementError> {
    project(column_major_intrinsics(intrinsics.row_major), point_camera)
        .ok_or(PlacementError::BehindCamera)
}

/// Model for the local-coordinate axes, converted from the OpenCV camera frame
/// to trd's GL camera frame. Use this for a `CoordinateAxes` drawable.
///
/// The red/green arms are the quad's own half-edges, not the orthonormalised
/// `e1`/`e2`, so they lie **on** the reconstructed quad edges — the gizmo the
/// Python reference draws (`--axes-local` over its placement-quad model, whose
/// columns are `r1/2`, `r2/2`, `n/2`). It is also the basis the in-plane
/// offsets are expressed in, so "move along green" matches what is on screen.
/// Orthonormalising dumps the quad's whole non-squareness onto the green arm:
/// on the FIBA tail, where the reconstructed edges close to 75°, `e2` swings
/// ~14° off the edge it is supposed to name.
pub fn quad_axes_model(frame: QuadFrame) -> Matrix4 {
    let model_camera = [
        frame.half_edge1[0],
        frame.half_edge1[1],
        frame.half_edge1[2],
        0.0,
        frame.half_edge2[0],
        frame.half_edge2[1],
        frame.half_edge2[2],
        0.0,
        frame.e3[0] * frame.axis_length,
        frame.e3[1] * frame.axis_length,
        frame.e3[2] * frame.axis_length,
        0.0,
        frame.origin_camera[0],
        frame.origin_camera[1],
        frame.origin_camera[2],
        1.0,
    ];
    let cv_to_gl = [
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    Matrix4::from_cols_array(&cv_to_gl) * Matrix4::from_cols_array(&model_camera)
}

/// Model whose unit XY square maps exactly to the reconstructed quad corners.
pub fn quad_outline_model(frame: QuadFrame) -> Matrix4 {
    let model_camera = [
        frame.half_edge1[0],
        frame.half_edge1[1],
        frame.half_edge1[2],
        0.0,
        frame.half_edge2[0],
        frame.half_edge2[1],
        frame.half_edge2[2],
        0.0,
        frame.e3[0] * frame.axis_length,
        frame.e3[1] * frame.axis_length,
        frame.e3[2] * frame.axis_length,
        0.0,
        frame.origin_camera[0],
        frame.origin_camera[1],
        frame.origin_camera[2],
        1.0,
    ];
    let cv_to_gl = [
        1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    Matrix4::from_cols_array(&cv_to_gl) * Matrix4::from_cols_array(&model_camera)
}

fn homography_unit_square_to_quad(quad: [[f32; 2]; 4]) -> Result<[f32; 9], PlacementError> {
    let [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = quad;
    let dx1 = x1 - x2;
    let dx2 = x3 - x2;
    let dx3 = x0 - x1 + x2 - x3;
    let dy1 = y1 - y2;
    let dy2 = y3 - y2;
    let dy3 = y0 - y1 + y2 - y3;
    let denominator = dx1 * dy2 - dx2 * dy1;
    if denominator.abs() <= 1e-8 {
        return Err(PlacementError::DegenerateQuad);
    }
    let g = (dx3 * dy2 - dx2 * dy3) / denominator;
    let h = (dx1 * dy3 - dx3 * dy1) / denominator;
    Ok([
        x1 - x0 + g * x1,
        y1 - y0 + g * y1,
        g,
        x3 - x0 + h * x3,
        y3 - y0 + h * y3,
        h,
        x0,
        y0,
        1.0,
    ])
}

fn column_major_intrinsics(row: [f32; 9]) -> [f32; 9] {
    [
        row[0], row[3], row[6], row[1], row[4], row[7], row[2], row[5], row[8],
    ]
}

fn inverse3(m: [f32; 9]) -> Option<[f32; 9]> {
    let determinant = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[3] * (m[1] * m[8] - m[2] * m[7])
        + m[6] * (m[1] * m[5] - m[2] * m[4]);
    if determinant.abs() <= 1e-8 {
        return None;
    }
    let inv = 1.0 / determinant;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv,
        (m[2] * m[7] - m[1] * m[8]) * inv,
        (m[1] * m[5] - m[2] * m[4]) * inv,
        (m[5] * m[6] - m[3] * m[8]) * inv,
        (m[0] * m[8] - m[2] * m[6]) * inv,
        (m[2] * m[3] - m[0] * m[5]) * inv,
        (m[3] * m[7] - m[4] * m[6]) * inv,
        (m[1] * m[6] - m[0] * m[7]) * inv,
        (m[0] * m[4] - m[1] * m[3]) * inv,
    ])
}

fn mat3_mul(m: [f32; 9], n: [f32; 9]) -> [f32; 9] {
    let mut result = [0.0; 9];
    for column in 0..3 {
        for row in 0..3 {
            result[column * 3 + row] = m[row] * n[column * 3]
                + m[3 + row] * n[column * 3 + 1]
                + m[6 + row] * n[column * 3 + 2];
        }
    }
    result
}

fn column3(m: [f32; 9], column: usize) -> [f32; 3] {
    [m[column * 3], m[column * 3 + 1], m[column * 3 + 2]]
}

fn project(k: [f32; 9], point: [f32; 3]) -> Option<[f32; 2]> {
    let projected = [
        k[0] * point[0] + k[3] * point[1] + k[6] * point[2],
        k[1] * point[0] + k[4] * point[1] + k[7] * point[2],
        k[2] * point[0] + k[5] * point[1] + k[8] * point[2],
    ];
    (projected[2].abs() > 1e-8).then(|| [projected[0] / projected[2], projected[1] / projected[2]])
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn length(value: [f32; 3]) -> f32 {
    dot(value, value).sqrt()
}

fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let length = length(value);
    (length > 1e-8).then(|| scale3(value, 1.0 / length))
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(value: [f32; 3], scale: f32) -> [f32; 3] {
    [value[0] * scale, value[1] * scale, value[2] * scale]
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn rectified_unit_square_recovers_orthonormal_frame() {
        let k = CameraIntrinsics {
            row_major: [1000.0, 0.0, 960.0, 0.0, 1000.0, 540.0, 0.0, 0.0, 1.0],
        };
        let quad = PlacementQuad {
            points_px: [
                [960.0, 540.0],
                [1160.0, 540.0],
                [1160.0, 740.0],
                [960.0, 740.0],
            ],
        };
        let frame = quad_frame(k, quad).unwrap();
        assert_relative_eq!(length(frame.e1), 1.0, epsilon = 1e-5);
        assert_relative_eq!(length(frame.e2), 1.0, epsilon = 1e-5);
        assert_relative_eq!(length(frame.e3), 1.0, epsilon = 1e-5);
        assert_relative_eq!(dot(frame.e1, frame.e2), 0.0, epsilon = 1e-5);
        assert_relative_eq!(dot(frame.e1, frame.e3), 0.0, epsilon = 1e-5);
        assert_relative_eq!(dot(frame.e2, frame.e3), 0.0, epsilon = 1e-5);
        assert_relative_eq!(
            dot(cross(frame.e1, frame.e2), frame.e3),
            1.0,
            epsilon = 1e-5
        );
    }

    #[test]
    fn identity_edit_places_a_finite_model() {
        let frame = QuadFrame {
            origin_camera: [0.0, 0.0, 5.0],
            e1: [1.0, 0.0, 0.0],
            e2: [0.0, 1.0, 0.0],
            e3: [0.0, 0.0, 1.0],
            half_edge1: [1.0, 0.0, 0.0],
            half_edge2: [0.0, 1.0, 0.0],
            axis_length: 1.0,
        };
        let model = placement_model(frame, LocalPlacement::default()).unwrap();
        assert!(model.to_cols_array().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn project_camera_accepts_row_major_intrinsics() {
        let projected = project_camera(
            CameraIntrinsics {
                row_major: [100.0, 0.0, 20.0, 0.0, 120.0, 30.0, 0.0, 0.0, 1.0],
            },
            [0.0, 0.0, 2.0],
        )
        .unwrap();
        assert_eq!(projected, [20.0, 30.0]);
    }

    #[test]
    fn fiba_frame_zero_matches_python_reference() {
        let frame = quad_frame(
            CameraIntrinsics {
                row_major: [4510.0986, 0.0, 960.0, 0.0, 4510.0986, 540.0, 0.0, 0.0, 1.0],
            },
            PlacementQuad {
                points_px: [
                    [752.1081, 541.8749],
                    [1292.765, 501.09924],
                    [1480.5444, 645.707],
                    [872.6903, 696.28595],
                ],
            },
        )
        .unwrap();
        let expected_origin = [0.23470776, 0.08677793, 7.632644];
        let expected_e1 = [0.974809, -0.07416263, 0.21035047];
        let expected_e2 = [-0.22217987, -0.24007191, 0.9449876];
        let expected_e3 = [-0.01958353, -0.96791804, -0.2505017];
        for (actual, expected) in frame.origin_camera.into_iter().zip(expected_origin) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
        for (actual, expected) in frame.e1.into_iter().zip(expected_e1) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
        for (actual, expected) in frame.e2.into_iter().zip(expected_e2) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
        for (actual, expected) in frame.e3.into_iter().zip(expected_e3) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
    }

    #[test]
    fn olympic_upper_can_frame_zero_matches_demo() {
        let frame = quad_frame(
            CameraIntrinsics {
                row_major: [4510.0986, 0.0, 960.0, 0.0, 4510.0986, 540.0, 0.0, 0.0, 1.0],
            },
            PlacementQuad {
                points_px: [
                    [752.1081, 541.8749],
                    [1292.765, 501.09924],
                    [1480.5444, 645.707],
                    [872.6903, 696.28595],
                ],
            },
        )
        .unwrap();
        let actual = placement_model(
            frame,
            LocalPlacement {
                offset_e1: 1.3,
                offset_e2: -1.7,
                size_factor: 0.24,
                ..Default::default()
            },
        )
        .unwrap()
        .to_cols_array();
        let expected = [
            0.11697708,
            0.008899516,
            -0.025242055,
            0.0,
            -0.002350024,
            0.11615017,
            0.030060204,
            0.0,
            0.026661586,
            -0.028808627,
            0.113398515,
            0.0,
            0.6685008,
            0.28248346,
            -8.546489,
            1.0,
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
    }

    #[test]
    fn yaw_matches_python_rotate_y_composition() {
        let frame = quad_frame(
            CameraIntrinsics {
                row_major: [4510.0986, 0.0, 960.0, 0.0, 4510.0986, 540.0, 0.0, 0.0, 1.0],
            },
            PlacementQuad {
                points_px: [
                    [752.1081, 541.8749],
                    [1292.765, 501.09924],
                    [1480.5444, 645.707],
                    [872.6903, 696.28595],
                ],
            },
        )
        .unwrap();
        let actual = placement_model(
            frame,
            LocalPlacement {
                offset_e1: 1.3,
                offset_e2: -1.7,
                size_factor: 0.24,
                yaw: 0.7,
                ..Default::default()
            },
        )
        .unwrap()
        .to_cols_array();
        // `C4 * (translate * [e1, e3, -e2] * rotate_y(0.7) * scale)`, from
        // examples/placement_quad_by_local_coord.py on FIBA shot 1 frame 0.
        let expected = [
            0.072_293_14,
            0.025365753,
            -0.092_359_52,
            0.0,
            -0.0023500241,
            0.11615017,
            0.030060203,
            0.0,
            0.095_750_61,
            -0.016300828,
            0.070_470_59,
            0.0,
            0.668_500_8,
            0.28248345,
            -8.546_489,
            1.0,
        ];
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = 2e-4);
        }
    }

    #[test]
    fn axes_arms_lie_on_the_quad_edges() {
        // The FIBA geometry: `cross(r1, r2)` points into the floor, so the
        // tip-above-origin test flips `e3` and `e2 = e3 × e1` lands
        // *anti-parallel* to the second half-edge — 165° apart once the
        // reconstructed edges close to 75° on frames 200–221. An axes gizmo
        // built from `e2` therefore points back out of the quad it annotates.
        let edge2 = (180.0f32 + 75.0).to_radians();
        let frame = QuadFrame {
            origin_camera: [0.0, 0.0, 5.0],
            e1: [1.0, 0.0, 0.0],
            e2: [0.0, 1.0, 0.0],
            e3: [0.0, 0.0, 1.0],
            half_edge1: [0.5, 0.0, 0.0],
            half_edge2: [0.5 * edge2.cos(), 0.5 * edge2.sin(), 0.0],
            axis_length: 0.5,
        };
        assert!(
            dot(frame.e2, frame.half_edge2) < 0.0,
            "fixture must reproduce the reversed orthonormal arm"
        );
        let columns = quad_axes_model(frame).to_cols_array();
        let red = [columns[0], columns[1], columns[2]];
        let green = [columns[4], columns[5], columns[6]];
        // `cv_to_gl` negates y and z, so compare against the flipped half-edges.
        for (actual, expected) in red.into_iter().zip([0.5, 0.0, 0.0]) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
        for (actual, expected) in
            green
                .into_iter()
                .zip([0.5 * edge2.cos(), -0.5 * edge2.sin(), 0.0])
        {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
        // Same frame as the quad outline, which is what the reference draws.
        for (axes, outline) in columns
            .into_iter()
            .zip(quad_outline_model(frame).to_cols_array())
        {
            assert_relative_eq!(axes, outline, epsilon = 1e-6);
        }
    }

    #[test]
    fn yaw_keeps_the_basis_a_rotation() {
        let frame = QuadFrame {
            origin_camera: [0.0, 0.0, 5.0],
            e1: [1.0, 0.0, 0.0],
            e2: [0.0, 1.0, 0.0],
            e3: [0.0, 0.0, 1.0],
            half_edge1: [1.0, 0.0, 0.0],
            half_edge2: [0.0, 1.0, 0.0],
            axis_length: 1.0,
        };
        // A sheared basis survives yaw = 0 and hides at small angles; ±45° is
        // where the mis-signed column degenerates outright.
        for yaw in [0.0, 0.3, std::f32::consts::FRAC_PI_4, 2.5, -1.9] {
            let size = 0.24;
            let columns = placement_model(
                frame,
                LocalPlacement {
                    size_factor: size,
                    yaw,
                    ..Default::default()
                },
            )
            .unwrap()
            .to_cols_array();
            let basis: [[f32; 3]; 3] = [
                [columns[0], columns[1], columns[2]],
                [columns[4], columns[5], columns[6]],
                [columns[8], columns[9], columns[10]],
            ];
            for axis in basis {
                assert_relative_eq!(length(axis), size, epsilon = 1e-6);
            }
            assert_relative_eq!(dot(basis[0], basis[1]), 0.0, epsilon = 1e-6);
            assert_relative_eq!(dot(basis[1], basis[2]), 0.0, epsilon = 1e-6);
            assert_relative_eq!(dot(basis[0], basis[2]), 0.0, epsilon = 1e-6);
            // Right-handed, so the mesh keeps its winding and normals.
            assert_relative_eq!(
                dot(cross(basis[0], basis[1]), basis[2]),
                size * size * size,
                epsilon = 1e-6
            );
        }
    }
}
