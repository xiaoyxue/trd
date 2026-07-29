//! Per-frame camera/model transform parameters and the projection
//! derived from camera intrinsics.

use crate::math::{Matrix4, Point3, Transform, Vector3};

/// Default clip near/far planes used when deriving a projection from camera
/// intrinsics `K`. The hello-triangle is authored on the `z = 0` plane, so the
/// exact values only need to bracket it; they are renderer constants until the
/// camera slice (#18) makes them configurable.
pub(crate) const DEFAULT_NEAR: f32 = 0.1;
pub(crate) const DEFAULT_FAR: f32 = 1000.0;
/// Per-frame transform parameters for the triangle.
///
/// The mesh vertices `p_i` are transformed by the full MVP chain
/// `clip = P · V · M · (p_i, 1)` in the vertex shader, where:
/// - **M** (model) is [`FrameParams::model`] if present, else the identity —
///   the default single-object placement when a frame carries no explicit
///   instanced draw list.
/// - **V** (view) is the camera-from-world transform, resolved (in precedence
///   order) from the **CV** pose [`FrameParams::pose`] (view = `inverse(pose)`),
///   else the **CG** look-at ([`FrameParams::eye`] + [`FrameParams::target`] or
///   [`FrameParams::direction`], with [`FrameParams::up`]), else identity.
/// - **P** (projection) is derived (in precedence order) from the **CV**
///   intrinsics [`FrameParams::k`] + viewport, else the **CG** perspective recipe
///   ([`FrameParams::fovy`] + [`FrameParams::aspect`]/[`FrameParams::znear`]/
///   [`FrameParams::zfar`]), else identity.
///
/// **CV wins over CG** (a well-formed stream carries only one form; mixing is
/// rejected at decode as a conflicting camera form). Any matrix/param that would
/// be identity is simply omitted (its column is absent), so a stream with no
/// camera columns has `P = V = I`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameParams {
    /// Optional explicit 4×4 **model** matrix, column-major (16 floats). Placed
    /// as the default single instance of mesh `0` when a frame carries no
    /// explicit instanced draw list; defaults to the identity when absent.
    pub model: Option<[f32; 16]>,
    /// Optional camera **intrinsics** `K` (**CV** form): a 3×3 pinhole matrix,
    /// column-major (9 floats). `Some` derives the projection; `None` falls back
    /// to the CG projection recipe or identity.
    pub k: Option<[f32; 9]>,
    /// Optional camera **pose** (**CV** form, world-from-camera): a 4×4 matrix,
    /// column-major (16 floats). The view matrix is its inverse; `None` falls
    /// back to the CG look-at or identity.
    pub pose: Option<[f32; 16]>,
    /// Optional camera **eye**/position (**CG** form): world-space `[x, y, z]`.
    pub eye: Option<[f32; 3]>,
    /// Optional CG look-at **target** point: world-space `[x, y, z]`. Takes
    /// precedence over [`FrameParams::direction`] when both are present.
    pub target: Option<[f32; 3]>,
    /// Optional CG forward **direction** the camera looks along from `eye`
    /// (`target = eye + direction`); an alternative to [`FrameParams::target`].
    pub direction: Option<[f32; 3]>,
    /// Optional CG **up** vector; defaults to `+Y` when absent.
    pub up: Option<[f32; 3]>,
    /// Optional CG vertical **field of view** in radians.
    pub fovy: Option<f32>,
    /// Optional CG **aspect** ratio (width/height); defaults to the viewport's.
    pub aspect: Option<f32>,
    /// Optional CG near clip plane; defaults to [`DEFAULT_NEAR`].
    pub znear: Option<f32>,
    /// Optional CG far clip plane; defaults to [`DEFAULT_FAR`].
    pub zfar: Option<f32>,
}

/// A malformed camera specification detected at decode time. Mapped by each
/// decoder to its stream/protocol error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraFormError {
    /// Both the CV form (`k`/`pose`) and the CG form (`eye`/`target`/
    /// `direction`/`fovy`) are present; a stream must use exactly one.
    Conflicting,
    /// The CG form is incomplete: an `eye` without a look `target`/`direction`,
    /// or a look `target`/`direction` without an `eye`.
    Incomplete,
}

/// The render target's pixel dimensions, needed to turn pixel-space camera
/// intrinsics `K` into a clip-space projection.
///
/// The viewport is a **size** (not a matrix baked into the MVP): it supplies the
/// pixel units that `K`'s `fx,fy,cx,cy` live in and the `aspect` ratio for a
/// projection. The NDC→pixel mapping (including the y-flip) is applied by the
/// render target / readback, matching the shipped pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

impl Viewport {
    /// The width/height aspect ratio (`>= 1` for landscape), guarding a zero
    /// height by treating each dimension as at least one pixel.
    #[inline]
    pub fn aspect(self) -> f32 {
        self.width.max(1) as f32 / self.height.max(1) as f32
    }
}

impl FrameParams {
    /// The identity transform: no model, no camera.
    pub const IDENTITY: FrameParams = FrameParams {
        model: None,
        k: None,
        pose: None,
        eye: None,
        target: None,
        direction: None,
        up: None,
        fovy: None,
        aspect: None,
        znear: None,
        zfar: None,
    };

    /// The effective 4×4 model matrix: the explicit [`FrameParams::model`] if
    /// present, else the identity. Used by front-ends to place the default single
    /// instance of mesh 0 when a frame carries no explicit instanced draw list.
    pub fn model_matrix(&self) -> Matrix4 {
        match self.model {
            Some(cols) => Matrix4::from_cols_array(&cols),
            None => Matrix4::IDENTITY,
        }
    }

    /// The view matrix `camera-from-world`, resolved in precedence order:
    /// **CV** `inverse(pose)`, else **CG** look-at (`eye` → `target` or
    /// `eye + direction`, with `up` defaulting to `+Y`), else identity.
    pub(crate) fn view_matrix(&self) -> Matrix4 {
        // CV form wins over CG.
        if let Some(cols) = self.pose {
            return Matrix4::from_cols_array(&cols).inverse();
        }
        if let Some(eye) = self.eye {
            let eye = Point3::new(eye[0], eye[1], eye[2]);
            let up = self
                .up
                .map(|u| Vector3::new(u[0], u[1], u[2]))
                .unwrap_or(Vector3::Y);
            // A look-at `target` takes precedence over a forward `direction`.
            let target = if let Some(t) = self.target {
                Point3::new(t[0], t[1], t[2])
            } else if let Some(d) = self.direction {
                eye + Vector3::new(d[0], d[1], d[2])
            } else {
                // Incomplete CG form (rejected at decode); be lenient here.
                return Matrix4::IDENTITY;
            };
            return Transform::look_at_rh(eye, target, up).matrix();
        }
        Matrix4::IDENTITY
    }

    /// The projection matrix, resolved in precedence order: **CV** intrinsics
    /// `K` + viewport, else **CG** perspective (`fovy`, `aspect` defaulting to
    /// the viewport's, `znear`/`zfar` defaulting to [`DEFAULT_NEAR`]/
    /// [`DEFAULT_FAR`]), else identity.
    pub(crate) fn projection_matrix(&self, viewport: Viewport) -> Matrix4 {
        if let Some(k) = self.k {
            return projection_from_intrinsics(k, viewport);
        }
        if let Some(fovy) = self.fovy {
            let aspect = self.aspect.unwrap_or_else(|| viewport.aspect());
            let near = self.znear.unwrap_or(DEFAULT_NEAR);
            let far = self.zfar.unwrap_or(DEFAULT_FAR);
            return Transform::perspective_rh(fovy, aspect, near, far).matrix();
        }
        Matrix4::IDENTITY
    }

    /// Validates the camera specification: exactly one of the CV (`k`/`pose`)
    /// and CG (`eye`/`target`/`direction`/`fovy`) forms, and a complete CG
    /// look-at (`eye` iff a look `target`/`direction`). A stream with no camera
    /// columns (all `None`) is valid (identity camera).
    pub(crate) fn check_camera_form(&self) -> Result<(), CameraFormError> {
        let cv = self.k.is_some() || self.pose.is_some();
        let look = self.target.is_some() || self.direction.is_some();
        let cg = self.eye.is_some() || look || self.fovy.is_some();
        if cv && cg {
            return Err(CameraFormError::Conflicting);
        }
        if self.eye.is_some() != look {
            return Err(CameraFormError::Incomplete);
        }
        Ok(())
    }

    /// The camera-only transform `P · V` for a given viewport, used by the
    /// instanced mesh path where each drawn instance supplies its own model
    /// matrix (`clip = P · V · M · p`).
    pub(crate) fn view_proj_matrix(&self, viewport: Viewport) -> Matrix4 {
        self.projection_matrix(viewport) * self.view_matrix()
    }

    /// The camera's **world-space position** — the translation of the inverse
    /// view (`world-from-camera`) matrix. Needed by the Disney PBR path for the
    /// view vector `V` (and environment reflection). Identity view ⇒ origin.
    pub(crate) fn camera_position(&self) -> [f32; 3] {
        let cols = self.view_matrix().inverse().to_cols_array();
        [cols[12], cols[13], cols[14]]
    }
}

/// Builds a right-handed, wgpu-clip-space (`z ∈ [0, 1]`) perspective projection
/// from a pinhole intrinsics matrix `K` (column-major: `fx = k[0]`, skew
/// `s = k[3]`, `fy = k[4]`, `cx = k[6]`, `cy = k[7]`) and the target viewport.
///
/// Conventions (to be validated visually / refined in the camera slice #18):
/// `K` shares NDC orientation (x right, y up, camera looking down `-z`); a
/// centered principal point (`cx = W/2`, `cy = H/2`) with square pixels and no
/// skew reduces to [`glam::Mat4::perspective_rh`]. Skew `s` shears the
/// projection (couples camera-`y` into clip-`x`). `near`/`far` are
/// [`DEFAULT_NEAR`]/[`DEFAULT_FAR`]. This is the exact inverse of
/// [`crate::Camera::to_intrinsics`], so `K ⇄ projection` round-trips losslessly.
pub(crate) fn projection_from_intrinsics(k: [f32; 9], viewport: Viewport) -> Matrix4 {
    let fx = k[0];
    let s = k[3];
    let fy = k[4];
    let cx = k[6];
    let cy = k[7];
    let w = viewport.width.max(1) as f32;
    let h = viewport.height.max(1) as f32;
    let (n, f) = (DEFAULT_NEAR, DEFAULT_FAR);

    // Column-major: each row below is one column of the matrix.
    Matrix4::from_cols_array(&[
        2.0 * fx / w,
        0.0,
        0.0,
        0.0,
        2.0 * s / w,
        2.0 * fy / h,
        0.0,
        0.0,
        2.0 * cx / w - 1.0,
        2.0 * cy / h - 1.0,
        f / (n - f),
        -1.0,
        0.0,
        0.0,
        (f * n) / (n - f),
        0.0,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Point3, Transform};
    use approx::assert_abs_diff_eq;
    use glam::{Mat4, Vec3, Vec4};

    #[test]
    fn identity_params_produce_identity_model() {
        assert_eq!(FrameParams::IDENTITY.model_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn explicit_model_used_verbatim() {
        // A `model` column value is used verbatim.
        let cols: [f32; 16] = std::array::from_fn(|i| i as f32 + 1.0);
        let params = FrameParams {
            model: Some(cols),
            ..FrameParams::IDENTITY
        };
        assert_eq!(params.model_matrix(), Matrix4::from_cols_array(&cols));
    }

    #[test]
    fn absent_model_falls_back_to_identity() {
        // A frame with no `model` column places mesh 0 at the identity.
        let params = FrameParams {
            model: None,
            ..FrameParams::IDENTITY
        };
        assert_eq!(params.model_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn view_matrix_is_pose_inverse() {
        let pose = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))
            * Mat4::from_rotation_y(0.7)
            * Mat4::from_rotation_x(-0.3);
        let params = FrameParams {
            pose: Some(pose.to_cols_array()),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            pose.inverse(),
            epsilon = 1e-5
        );
        // No pose => identity view.
        assert_eq!(FrameParams::IDENTITY.view_matrix(), Matrix4::IDENTITY);
    }

    #[test]
    fn cg_view_matches_look_at() {
        // The CG `eye`/`target`/`up` form resolves to the same view matrix as a
        // direct look-at.
        let params = FrameParams {
            eye: Some([2.0, 3.0, 5.0]),
            target: Some([0.1, 0.2, 0.3]),
            up: Some([0.0, 1.0, 0.0]),
            fovy: Some(0.9),
            ..FrameParams::IDENTITY
        };
        let expected = Transform::look_at_rh(
            Point3::new(2.0, 3.0, 5.0),
            Point3::new(0.1, 0.2, 0.3),
            Vector3::Y,
        )
        .matrix();
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cg_direction_matches_target_at_eye_plus_direction() {
        // `direction` resolves to `target = eye + direction`; `up` defaults to +Y.
        let eye = [1.0, 2.0, 3.0];
        let dir = [0.0, 0.0, -1.0];
        let via_dir = FrameParams {
            eye: Some(eye),
            direction: Some(dir),
            ..FrameParams::IDENTITY
        };
        let via_target = FrameParams {
            eye: Some(eye),
            target: Some([1.0, 2.0, 2.0]),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            via_dir.view_matrix().into_inner(),
            via_target.view_matrix().into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cv_pose_wins_over_cg_view() {
        // Even if CG `eye`/`target` are present, a CV `pose` takes precedence for
        // the view matrix (a well-formed stream never mixes them; this pins the
        // resolution order).
        let pose = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)) * Mat4::from_rotation_y(0.4);
        let params = FrameParams {
            pose: Some(pose.to_cols_array()),
            eye: Some([9.0, 9.0, 9.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        assert_abs_diff_eq!(
            params.view_matrix().into_inner(),
            pose.inverse(),
            epsilon = 1e-5
        );
    }

    #[test]
    fn cg_projection_matches_perspective() {
        let viewport = Viewport {
            width: 800,
            height: 600,
        };
        let params = FrameParams {
            fovy: Some(0.9),
            aspect: Some(1.5),
            znear: Some(0.5),
            zfar: Some(50.0),
            eye: Some([0.0, 0.0, 1.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        let expected = Transform::perspective_rh(0.9, 1.5, 0.5, 50.0).matrix();
        assert_abs_diff_eq!(
            params.projection_matrix(viewport).into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn cg_projection_defaults_aspect_and_clip_planes() {
        // `aspect` defaults to the viewport's; `znear`/`zfar` to the renderer's.
        let viewport = Viewport {
            width: 800,
            height: 400,
        };
        let params = FrameParams {
            fovy: Some(0.8),
            eye: Some([0.0, 0.0, 1.0]),
            target: Some([0.0, 0.0, 0.0]),
            ..FrameParams::IDENTITY
        };
        let expected =
            Transform::perspective_rh(0.8, viewport.aspect(), DEFAULT_NEAR, DEFAULT_FAR).matrix();
        assert_abs_diff_eq!(
            params.projection_matrix(viewport).into_inner(),
            expected.into_inner(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn camera_form_validation() {
        // No camera columns: valid (identity camera).
        assert_eq!(FrameParams::IDENTITY.check_camera_form(), Ok(()));
        // CV form (k + pose): valid.
        assert_eq!(
            FrameParams {
                k: Some([0.0; 9]),
                pose: Some([0.0; 16]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // CG look-at (eye + target): valid.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                target: Some([0.0; 3]),
                fovy: Some(1.0),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // CG forward (eye + direction): valid.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                direction: Some([0.0, 0.0, -1.0]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Ok(())
        );
        // Mixing CV and CG: rejected.
        assert_eq!(
            FrameParams {
                k: Some([0.0; 9]),
                eye: Some([0.0; 3]),
                target: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Conflicting)
        );
        // `eye` without a look target/direction: incomplete.
        assert_eq!(
            FrameParams {
                eye: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Incomplete)
        );
        // Look `target` without an `eye`: incomplete.
        assert_eq!(
            FrameParams {
                target: Some([0.0; 3]),
                ..FrameParams::IDENTITY
            }
            .check_camera_form(),
            Err(CameraFormError::Incomplete)
        );
    }

    #[test]
    fn centered_square_intrinsics_match_glam_perspective() {
        // A centered principal point with square pixels must reduce to glam's
        // right-handed perspective (fov_y from fy).
        let viewport = Viewport {
            width: 800,
            height: 600,
        };
        let (w, h) = (viewport.width as f32, viewport.height as f32);
        let f = 500.0_f32; // fx = fy
        let k = [f, 0.0, 0.0, 0.0, f, 0.0, w / 2.0, h / 2.0, 1.0];

        let got = projection_from_intrinsics(k, viewport);
        let fov_y = 2.0 * (h / (2.0 * f)).atan();
        let expected = Mat4::perspective_rh(fov_y, w / h, DEFAULT_NEAR, DEFAULT_FAR);
        assert_abs_diff_eq!(got.into_inner(), expected, epsilon = 1e-4);
    }

    #[test]
    fn principal_axis_projects_to_principal_point() {
        // A camera-space point straight ahead (on the optical axis, -z) lands at
        // the principal point in NDC (0,0 for a centered K).
        let viewport = Viewport {
            width: 640,
            height: 480,
        };
        let (w, h) = (viewport.width as f32, viewport.height as f32);
        let k = [400.0, 0.0, 0.0, 0.0, 400.0, 0.0, w / 2.0, h / 2.0, 1.0];
        let p = projection_from_intrinsics(k, viewport);
        let clip = p.into_inner() * Vec4::new(0.0, 0.0, -5.0, 1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-5 && ndc.y.abs() < 1e-5, "ndc = {ndc:?}");
    }
}
