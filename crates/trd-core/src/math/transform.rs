//! [`Transform`] — an invertible affine/projective transform that knows how to
//! move each geometric kind correctly.
//!
//! It caches the inverse (decision 4) so [`Transform::transform_normal`] (which
//! needs `(M⁻¹)ᵀ`) and [`Transform::inverse`] are allocation-free and can't
//! silently recompute a near-singular inverse per call.

use super::aabb::Aabb3;
use super::linalg::{Matrix3, Matrix4, Normal3, Point3, Rotation, Vector3};
use glam::{Mat3, Mat4};

/// An invertible transform, stored as a matrix and its cached inverse.
///
/// Application methods respect the affine algebra:
/// - [`Self::transform_point`] — affine map, **no** perspective divide.
/// - [`Self::transform_vector`] — ignores translation (the linear part only).
/// - [`Self::transform_normal`] — inverse-transpose, stays perpendicular.
/// - [`Self::project_point`] — full projective map **with** perspective divide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    m: Matrix4,
    m_inv: Matrix4,
}

impl Default for Transform {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    /// The identity transform.
    pub const IDENTITY: Self = Self {
        m: Matrix4::IDENTITY,
        m_inv: Matrix4::IDENTITY,
    };

    /// Builds from a matrix, computing (and caching) its inverse.
    ///
    /// The inverse is glam's `Mat4::inverse` (garbage, not a panic, if `m` is
    /// singular). Use [`Self::from_matrix_checked`] to reject singular input.
    #[inline]
    pub fn from_matrix(m: Matrix4) -> Self {
        Self {
            m,
            m_inv: m.inverse(),
        }
    }

    /// Builds from a matrix, returning `None` if it is (near-)singular.
    #[inline]
    pub fn from_matrix_checked(m: Matrix4) -> Option<Self> {
        m.try_inverse().map(|m_inv| Self { m, m_inv })
    }

    /// Builds from a matrix and a known-correct inverse (no recomputation).
    ///
    /// The caller guarantees `m * m_inv == I`; used by [`Self::inverse`] and by
    /// constructors with a closed-form inverse.
    #[inline]
    fn from_pair(m: Matrix4, m_inv: Matrix4) -> Self {
        Self { m, m_inv }
    }

    /// A pure translation.
    #[inline]
    pub fn from_translation(v: Vector3) -> Self {
        Self::from_pair(Matrix4::from_translation(v), Matrix4::from_translation(-v))
    }

    /// A pure rotation.
    #[inline]
    pub fn from_rotation(r: Rotation) -> Self {
        Self::from_pair(
            Matrix4::from_rotation(r),
            Matrix4::from_rotation(r.inverse()),
        )
    }

    /// A pure (non-zero) scale.
    #[inline]
    pub fn from_scale(v: Vector3) -> Self {
        Self::from_matrix(Matrix4::from_scale(v))
    }

    /// The standard TRS model transform: scale, then rotate, then translate.
    #[inline]
    pub fn from_scale_rotation_translation(
        scale: Vector3,
        rotation: Rotation,
        translation: Vector3,
    ) -> Self {
        Self::from_matrix(Matrix4::from_glam(Mat4::from_scale_rotation_translation(
            scale.into_inner(),
            rotation.into_inner(),
            translation.into_inner(),
        )))
    }

    /// A right-handed look-at **view** transform (`z ∈ [0,1]` convention).
    #[inline]
    pub fn look_at_rh(eye: Point3, target: Point3, up: Vector3) -> Self {
        Self::from_matrix(Matrix4::from_glam(Mat4::look_at_rh(
            eye.into_inner(),
            target.into_inner(),
            up.into_inner(),
        )))
    }

    /// A right-handed perspective **projection** with reverse-infinite-free,
    /// `z ∈ [0,1]` clip depth (wgpu). `fov_y_radians` is the vertical FOV.
    #[inline]
    pub fn perspective_rh(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Self {
        Self::from_matrix(Matrix4::from_glam(Mat4::perspective_rh(
            fov_y_radians,
            aspect,
            near,
            far,
        )))
    }

    /// A right-handed orthographic **projection** with `z ∈ [0,1]` clip depth.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn orthographic_rh(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self::from_matrix(Matrix4::from_glam(Mat4::orthographic_rh(
            left, right, bottom, top, near, far,
        )))
    }

    /// The forward matrix.
    #[inline]
    pub fn matrix(self) -> Matrix4 {
        self.m
    }

    /// The cached inverse matrix.
    #[inline]
    pub fn inverse_matrix(self) -> Matrix4 {
        self.m_inv
    }

    /// The inverse transform (swaps the cached pair — no matrix inverse).
    #[inline]
    pub fn inverse(self) -> Self {
        Self::from_pair(self.m_inv, self.m)
    }

    /// Composes so that `self` is applied first: `self.then(next) == next * self`.
    #[inline]
    pub fn then(self, next: Self) -> Self {
        Self::from_pair(next.m * self.m, self.m_inv * next.m_inv)
    }

    /// Transforms a point (affine, **no** perspective divide).
    #[inline]
    pub fn transform_point(self, p: Point3) -> Point3 {
        Point3::from_glam(self.m.into_inner().transform_point3(p.into_inner()))
    }

    /// Transforms a free direction (ignores translation).
    #[inline]
    pub fn transform_vector(self, v: Vector3) -> Vector3 {
        Vector3::from_glam(self.m.into_inner().transform_vector3(v.into_inner()))
    }

    /// Transforms a normal by the inverse-transpose of the linear part.
    ///
    /// The result is **not** re-normalized (non-uniform scale changes length);
    /// call [`Normal3::normalize_or_zero`] if you need a unit normal.
    #[inline]
    pub fn transform_normal(self, n: Normal3) -> Normal3 {
        let linear_inv_t: Mat3 = Mat3::from_mat4(self.m_inv.into_inner()).transpose();
        Normal3::from_glam(linear_inv_t * n.into_inner())
    }

    /// Projects a point through the full transform **with** perspective divide.
    #[inline]
    pub fn project_point(self, p: Point3) -> Point3 {
        Point3::from_glam(self.m.into_inner().project_point3(p.into_inner()))
    }

    /// The tight axis-aligned box enclosing the transformed input box.
    ///
    /// Transforms the eight corners with [`Self::transform_point`] (affine, no
    /// perspective divide) and re-encloses them — correct for the affine
    /// model/view transforms it is used with. An empty input stays empty.
    #[inline]
    pub fn transform_aabb(self, aabb: Aabb3) -> Aabb3 {
        if aabb.is_empty() {
            return Aabb3::EMPTY;
        }
        Aabb3::from_points(aabb.corners().map(|c| self.transform_point(c)))
    }

    /// The upper-left 3×3 linear part.
    #[inline]
    pub fn linear(self) -> Matrix3 {
        self.m.to_mat3()
    }

    /// The column-major `[f32; 16]` of the forward matrix (GPU upload form).
    #[inline]
    pub fn to_cols_array(self) -> [f32; 16] {
        self.m.to_cols_array()
    }
}

impl From<Matrix4> for Transform {
    #[inline]
    fn from(m: Matrix4) -> Self {
        Self::from_matrix(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Aabb3, EPSILON};
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use proptest::prelude::*;

    fn finite() -> impl Strategy<Value = f32> {
        -100.0f32..100.0
    }
    fn vec3() -> impl Strategy<Value = Vector3> {
        (finite(), finite(), finite()).prop_map(|(x, y, z)| Vector3::new(x, y, z))
    }
    fn point3() -> impl Strategy<Value = Point3> {
        (finite(), finite(), finite()).prop_map(|(x, y, z)| Point3::new(x, y, z))
    }
    fn aabb3() -> impl Strategy<Value = Aabb3> {
        (point3(), point3()).prop_map(|(a, b)| Aabb3::from_corners(a, b))
    }

    proptest! {
        #[test]
        fn identity_is_a_no_op(p in point3(), v in vec3()) {
            let t = Transform::IDENTITY;
            assert_abs_diff_eq!(t.transform_point(p).into_inner(), p.into_inner(), epsilon = EPSILON);
            assert_abs_diff_eq!(t.transform_vector(v).into_inner(), v.into_inner(), epsilon = EPSILON);
        }

        #[test]
        fn vector_ignores_translation(t_vec in vec3(), v in vec3()) {
            // A pure translation must leave a free direction unchanged.
            let t = Transform::from_translation(t_vec);
            assert_abs_diff_eq!(t.transform_vector(v).into_inner(), v.into_inner(), epsilon = EPSILON);
        }

        #[test]
        fn point_respects_translation(t_vec in vec3(), p in point3()) {
            let t = Transform::from_translation(t_vec);
            assert_relative_eq!(
                t.transform_point(p).into_inner(),
                (p + t_vec).into_inner(),
                epsilon = EPSILON,
                max_relative = 1e-5
            );
        }

        #[test]
        fn then_composition_matches_matrix_product(a_t in vec3(), b_t in vec3(), p in point3()) {
            let a = Transform::from_translation(a_t);
            let b = Transform::from_rotation(Rotation::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0), 0.5,
            )).then(Transform::from_translation(b_t));
            // a.then(b) applies a first, then b.
            let composed = a.then(b);
            let step = b.transform_point(a.transform_point(p));
            assert_abs_diff_eq!(
                composed.transform_point(p).into_inner(),
                step.into_inner(),
                epsilon = 1e-3
            );
        }

        #[test]
        fn inverse_undoes_transform(t_vec in vec3(), p in point3()) {
            let t = Transform::from_scale(Vector3::new(2.0, 3.0, 0.5))
                .then(Transform::from_translation(t_vec));
            let back = t.inverse().transform_point(t.transform_point(p));
            assert_abs_diff_eq!(back.into_inner(), p.into_inner(), epsilon = 1e-3);
        }

        #[test]
        fn transform_aabb_encloses_transformed_corners(
            b in aabb3(),
            angle in -3.0f32..3.0,
            t_vec in vec3(),
        ) {
            // The transformed box must enclose every transformed corner (with a
            // small tolerance for the corner-recompute rounding).
            let t = Transform::from_rotation(Rotation::from_axis_angle(
                Vector3::new(0.3, 1.0, -0.5), angle,
            )).then(Transform::from_translation(t_vec));
            let out = t.transform_aabb(b);
            for c in b.corners() {
                prop_assert!(out.expanded(1e-3).contains_point(t.transform_point(c)));
            }
        }
    }

    #[test]
    fn transform_aabb_translation_is_exact() {
        // A pure translation shifts both corners by the same vector.
        let b = Aabb3::from_corners(Point3::new(-1.0, -2.0, -3.0), Point3::new(1.0, 2.0, 3.0));
        let t = Transform::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let out = t.transform_aabb(b);
        assert_abs_diff_eq!(
            out.min().into_inner(),
            Point3::new(9.0, 18.0, 27.0).into_inner(),
            epsilon = EPSILON
        );
        assert_abs_diff_eq!(
            out.max().into_inner(),
            Point3::new(11.0, 22.0, 33.0).into_inner(),
            epsilon = EPSILON
        );
        // Empty in, empty out.
        assert!(t.transform_aabb(Aabb3::EMPTY).is_empty());
    }

    #[test]
    fn normal_stays_perpendicular_under_nonuniform_scale() {
        // Two orthogonal tangents and their normal; under a non-uniform scale the
        // naive (linear) transform of the normal is NOT perpendicular, but the
        // inverse-transpose transform is.
        let scale = Transform::from_scale(Vector3::new(2.0, 1.0, 1.0));
        let tangent = Vector3::new(1.0, 1.0, 0.0);
        let normal = Normal3::new(1.0, -1.0, 0.0); // ⟂ tangent

        let t2 = scale.transform_vector(tangent);
        let n2 = scale.transform_normal(normal);
        assert!(
            n2.dot(t2).abs() < 1e-5,
            "normal not perpendicular: {}",
            n2.dot(t2)
        );
    }

    #[test]
    fn project_point_applies_perspective_divide() {
        // A perspective projection of an on-axis point lands at the principal
        // point; project_point divides by w, transform_point would not.
        let proj = Transform::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let ndc = proj.project_point(Point3::new(0.0, 0.0, -5.0));
        assert!(
            ndc.x().abs() < 1e-6 && ndc.y().abs() < 1e-6,
            "ndc = {ndc:?}"
        );
        assert!((0.0..=1.0).contains(&ndc.z()), "z in [0,1]: {}", ndc.z());
    }

    #[test]
    fn from_matrix_checked_rejects_singular() {
        assert!(Transform::from_matrix_checked(Matrix4::from_scale(Vector3::ZERO)).is_none());
        assert!(Transform::from_matrix_checked(Matrix4::IDENTITY).is_some());
    }

    #[test]
    fn look_at_places_eye_at_origin() {
        let view = Transform::look_at_rh(Point3::new(0.0, 0.0, 5.0), Point3::ORIGIN, Vector3::Y);
        // The eye maps to the view-space origin.
        let eye_in_view = view.transform_point(Point3::new(0.0, 0.0, 5.0));
        assert_abs_diff_eq!(eye_in_view.into_inner(), glam::Vec3::ZERO, epsilon = 1e-5);
    }
}
