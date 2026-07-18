//! [`Camera`] — one camera described in two **equivalent** ways (#43).
//!
//! trd accepts a camera from either world:
//!
//! - **Graphics / GL pipeline:** a **view** transform (world→camera), a
//!   **projection** (camera→clip, `z ∈ [0, 1]`), and the **viewport** (the pixel
//!   size the image is rasterized into).
//! - **Computer-vision / pinhole:** camera **intrinsics `K`** (a 3×3 pinhole
//!   matrix) plus an extrinsic **pose `[R | t]`** (world-from-camera).
//!
//! The two forms describe the *same* camera and convert losslessly (for a
//! pinhole with matching viewport and clip-depth range):
//!
//! ```text
//! graphics:  clip = Projection · View · X_world          (pixel = viewport(NDC))
//! CV:        [u v 1]ᵀ  ∝  K · [R | t] · X_world
//! ```
//!
//! [`Camera::from_gl`] and [`Camera::from_cv`] build the same [`Camera`], and
//! [`Camera::to_intrinsics`] / [`Camera::to_pose`] recover the CV form. The GL
//! triple `(view, projection, viewport)` is the single stored source of truth;
//! the `K`↔projection mapping is exactly the renderer's
//! [`crate::render`]`::projection_from_intrinsics` (which this module inverts),
//! so a camera wired through [`FrameParams`](crate::FrameParams)'s `k` + `pose`
//! columns renders identically whether it was authored in GL or CV form.
//!
//! Conventions (repo-wide, see [`crate::math`]): right-handed world, **+Y up**,
//! wgpu clip `z ∈ [0, 1]`, column-major matrices, radians. The NDC→pixel step
//! (including the y-flip) is applied by the render target, not baked into the
//! MVP — so the [`Viewport`] here is a *size*, carrying the pixel units `K`
//! lives in and the `aspect` ratio.

use crate::math::{Aabb3, Point3, Transform, Vector3, EPSILON};
use crate::render::{projection_from_intrinsics, Viewport};

/// The default vertical field of view for the framing camera: `45°`.
pub const DEFAULT_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// The default view direction for [`Camera::fit_aabb`]: `+Z`, i.e. the eye sits
/// on the `+Z` side looking toward `-Z` (a front view, camera down `-z`).
pub const DEFAULT_VIEW_DIR: Vector3 = Vector3::Z;

/// The default padding applied to the fit distance so the framed bounds don't
/// touch the image edge (`1.1` ⇒ ~10% margin).
pub const DEFAULT_FIT_MARGIN: f32 = 1.1;

/// A camera stored as the graphics triple `(view, projection, viewport)`, with
/// lossless conversion to/from the CV form (`K`, pose `[R | t]`).
///
/// Build one with [`Camera::from_gl`], [`Camera::from_cv`], or the framing
/// helper [`Camera::fit_aabb`]. Recover the CV form with [`Camera::to_intrinsics`]
/// and [`Camera::to_pose`]; feed a renderer with [`Camera::view_projection`] (the
/// clip-from-world `P · V`) or the `k`/`pose` pair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// World→camera rigid transform.
    view: Transform,
    /// Camera→clip projection (`z ∈ [0, 1]`).
    projection: Transform,
    /// Pixel size the image is rasterized into (units for `K`, source of aspect).
    viewport: Viewport,
}

impl Camera {
    /// Builds a camera from the **graphics** form: a `view` (world→camera), a
    /// `projection` (camera→clip), and the target `viewport`.
    #[inline]
    pub fn from_gl(view: Transform, projection: Transform, viewport: Viewport) -> Self {
        Self {
            view,
            projection,
            viewport,
        }
    }

    /// Builds a camera from the **computer-vision** form: pinhole intrinsics
    /// `K` (column-major `[fx, 0, 0, s, fy, 0, cx, cy, 1]`) and an extrinsic
    /// `pose` (**world-from-camera**; the view is its inverse).
    ///
    /// The projection is derived from `K` and the viewport exactly as the
    /// renderer does, so this is the inverse of [`Camera::to_intrinsics`] /
    /// [`Camera::to_pose`] (up to the projection's fixed near/far).
    #[inline]
    pub fn from_cv(k: [f32; 9], pose: Transform, viewport: Viewport) -> Self {
        Self {
            view: pose.inverse(),
            projection: Transform::from_matrix(projection_from_intrinsics(k, viewport)),
            viewport,
        }
    }

    /// The stored world→camera **view** transform.
    #[inline]
    pub fn view(self) -> Transform {
        self.view
    }

    /// The stored camera→clip **projection** transform.
    #[inline]
    pub fn projection(self) -> Transform {
        self.projection
    }

    /// The target **viewport** (pixel size).
    #[inline]
    pub fn viewport(self) -> Viewport {
        self.viewport
    }

    /// The clip-from-world transform `P · V` (apply the model matrix on the
    /// right for the full MVP). This is what feeds the GPU uniform.
    #[inline]
    pub fn view_projection(self) -> Transform {
        self.view.then(self.projection)
    }

    /// The extrinsic **pose** (world-from-camera) — the inverse of the view.
    #[inline]
    pub fn to_pose(self) -> Transform {
        self.view.inverse()
    }

    /// The pinhole **intrinsics `K`** recovered from the projection and viewport
    /// (column-major `[fx, 0, 0, s, fy, 0, cx, cy, 1]`).
    ///
    /// Exact inverse of the renderer's `projection_from_intrinsics`: reads back
    /// `fx, fy` (focal, in pixels), `s` (skew), and the principal point
    /// `cx, cy`. The projection's near/far are not represented in `K`.
    pub fn to_intrinsics(self) -> [f32; 9] {
        let m = self.projection.to_cols_array();
        let w = self.viewport.width.max(1) as f32;
        let h = self.viewport.height.max(1) as f32;
        let fx = m[0] * w / 2.0;
        let s = m[4] * w / 2.0;
        let fy = m[5] * h / 2.0;
        let cx = (m[8] + 1.0) * w / 2.0;
        let cy = (m[9] + 1.0) * h / 2.0;
        [fx, 0.0, 0.0, s, fy, 0.0, cx, cy, 1.0]
    }

    /// A **default framing camera** that fits `aabb` (a bounding box, typically
    /// the origin-centered mesh of #37) into the `viewport` from the
    /// [`DEFAULT_VIEW_DIR`] with the [`DEFAULT_FOV_Y`] and [`DEFAULT_FIT_MARGIN`].
    #[inline]
    pub fn fit_aabb(aabb: Aabb3, viewport: Viewport) -> Self {
        Self::fit_aabb_with(
            aabb,
            viewport,
            DEFAULT_FOV_Y,
            DEFAULT_VIEW_DIR,
            Vector3::Y,
            DEFAULT_FIT_MARGIN,
        )
    }

    /// A framing camera with an explicit `fov_y`, view `direction` (eye =
    /// target + dir·distance), `up`, and fit `margin`.
    ///
    /// Frames the box's bounding sphere so it fits both image dimensions
    /// (accounting for `aspect`): `distance = radius / sin(min_half_fov) ·
    /// margin`, with `near`/`far` bracketing the sphere. An empty `aabb` is
    /// treated as a unit box at the origin.
    pub fn fit_aabb_with(
        aabb: Aabb3,
        viewport: Viewport,
        fov_y: f32,
        direction: Vector3,
        up: Vector3,
        margin: f32,
    ) -> Self {
        let (center, radius) = if aabb.is_empty() {
            (Point3::ORIGIN, 1.0)
        } else {
            (aabb.center(), aabb.half_extents().length().max(EPSILON))
        };

        let aspect = viewport.aspect();
        let half_fov_y = 0.5 * fov_y;
        // Horizontal half-fov for the given aspect; fit the tighter axis.
        let half_fov_x = (aspect * half_fov_y.tan()).atan();
        let min_half_fov = half_fov_y.min(half_fov_x).max(EPSILON);
        let distance = (radius / min_half_fov.sin()) * margin;

        let dir = direction.normalize_or_zero();
        let dir = if dir.length_squared() > 0.0 {
            dir
        } else {
            DEFAULT_VIEW_DIR
        };
        let eye = center + dir * distance;

        // Bracket the sphere; keep `near` strictly positive.
        let near = (distance - radius).max(radius * 1.0e-2).max(EPSILON);
        let far = distance + radius;

        let view = Transform::look_at_rh(eye, center, up);
        let projection = Transform::perspective_rh(fov_y, aspect, near, far);
        Self::from_gl(view, projection, viewport)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Matrix4;
    use approx::assert_abs_diff_eq;

    fn viewport() -> Viewport {
        Viewport {
            width: 800,
            height: 600,
        }
    }

    // A representative non-trivial view (world→camera), from a look-at pose.
    fn sample_view() -> Transform {
        Transform::look_at_rh(
            Point3::new(2.0, 1.5, 4.0),
            Point3::new(0.2, -0.1, 0.0),
            Vector3::Y,
        )
    }

    // A pinhole K with an off-center principal point and mild skew.
    fn sample_k() -> [f32; 9] {
        let vp = viewport();
        let (w, h) = (vp.width as f32, vp.height as f32);
        [
            520.0,
            0.0,
            0.0,
            3.0, // skew
            510.0,
            0.0,
            w / 2.0 + 12.0, // cx off-center
            h / 2.0 - 8.0,  // cy off-center
            1.0,
        ]
    }

    #[test]
    fn intrinsics_round_trip_through_projection() {
        // K -> projection -> K must be the identity (incl. skew + off-center).
        let vp = viewport();
        let k = sample_k();
        let cam = Camera::from_cv(k, Transform::IDENTITY, vp);
        let k2 = cam.to_intrinsics();
        for (a, b) in k.iter().zip(k2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-3);
        }
    }

    #[test]
    fn gl_and_cv_forms_describe_the_same_camera() {
        // Build in GL form, convert to CV, rebuild, and require identical
        // clip coordinates for a spread of world points.
        let vp = viewport();
        let view = sample_view();
        let projection = Transform::from_matrix(projection_from_intrinsics(sample_k(), vp));
        let cam_gl = Camera::from_gl(view, projection, vp);

        let cam_cv = Camera::from_cv(cam_gl.to_intrinsics(), cam_gl.to_pose(), vp);

        let vp_gl = cam_gl.view_projection();
        let vp_cv = cam_cv.view_projection();
        for p in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, -0.5, 0.3),
            Point3::new(-0.7, 0.9, -1.2),
            Point3::new(2.0, 2.0, 2.0),
        ] {
            assert_abs_diff_eq!(
                vp_gl.project_point(p).into_inner(),
                vp_cv.project_point(p).into_inner(),
                epsilon = 1e-3
            );
        }
    }

    #[test]
    fn pose_is_the_view_inverse() {
        let vp = viewport();
        let cam = Camera::from_gl(sample_view(), Transform::IDENTITY, vp);
        assert_abs_diff_eq!(
            cam.to_pose().matrix().into_inner(),
            sample_view().inverse_matrix().into_inner(),
            epsilon = 1e-5
        );
        // from_cv(pose) then to_pose() recovers the pose.
        let pose = Transform::from_matrix(Matrix4::from_cols_array(
            &sample_view().inverse().to_cols_array(),
        ));
        let cam = Camera::from_cv(sample_k(), pose, vp);
        assert_abs_diff_eq!(
            cam.to_pose().matrix().into_inner(),
            pose.matrix().into_inner(),
            epsilon = 1e-4
        );
    }

    #[test]
    fn centered_square_intrinsics_recover_focal_from_fov() {
        // A centered perspective projection recovers fx = fy = h / (2 tan(fovy/2)).
        let vp = Viewport {
            width: 640,
            height: 640,
        };
        let fov_y = 1.0_f32;
        let cam = Camera::from_gl(
            Transform::IDENTITY,
            Transform::perspective_rh(fov_y, vp.aspect(), 0.1, 100.0),
            vp,
        );
        let k = cam.to_intrinsics();
        let expected_f = vp.height as f32 / (2.0 * (fov_y * 0.5).tan());
        assert_abs_diff_eq!(k[0], expected_f, epsilon = 1e-2); // fx
        assert_abs_diff_eq!(k[4], expected_f, epsilon = 1e-2); // fy
        assert_abs_diff_eq!(k[6], vp.width as f32 / 2.0, epsilon = 1e-2); // cx
        assert_abs_diff_eq!(k[7], vp.height as f32 / 2.0, epsilon = 1e-2); // cy
    }

    #[test]
    fn fit_aabb_frames_the_box_inside_ndc() {
        // Every corner of a centered box must land inside the NDC cube.
        let aabb = Aabb3::from_corners(Point3::new(-0.5, -0.8, -0.3), Point3::new(0.5, 0.8, 0.3));
        let vp = viewport();
        let cam = Camera::fit_aabb(aabb, vp);
        let vpm = cam.view_projection();
        for c in aabb.corners() {
            let ndc = vpm.project_point(c);
            assert!(
                ndc.x().abs() <= 1.0 + 1e-3 && ndc.y().abs() <= 1.0 + 1e-3,
                "corner {c:?} -> ndc {ndc:?} outside x/y"
            );
            assert!(
                (0.0 - 1e-3..=1.0 + 1e-3).contains(&ndc.z()),
                "corner {c:?} -> ndc z {} outside [0,1]",
                ndc.z()
            );
        }
    }

    #[test]
    fn fit_aabb_looks_at_the_center() {
        // The box center projects to the principal point (NDC origin in x,y).
        let aabb = Aabb3::from_corners(Point3::new(1.0, 2.0, 3.0), Point3::new(3.0, 4.0, 5.0));
        let vp = viewport();
        let cam = Camera::fit_aabb(aabb, vp);
        let ndc = cam.view_projection().project_point(aabb.center());
        assert!(
            ndc.x().abs() < 1e-4 && ndc.y().abs() < 1e-4,
            "ndc = {ndc:?}"
        );
    }

    #[test]
    fn fit_empty_aabb_is_finite() {
        let cam = Camera::fit_aabb(Aabb3::EMPTY, viewport());
        assert!(cam
            .view_projection()
            .to_cols_array()
            .iter()
            .all(|x| x.is_finite()));
    }
}
