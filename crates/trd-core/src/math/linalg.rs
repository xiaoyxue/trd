//! Typed linear-algebra newtypes over glam: [`Vector2`]/[`Vector3`]/[`Vector4`],
//! [`Point2`]/[`Point3`]/[`Point4`], [`Normal3`], [`Rotation`], [`Matrix3`],
//! [`Matrix4`].
//!
//! Inner fields are **private**: exposing `.0` would let callers bypass the
//! affine rules (e.g. `a.0 + b.0` to fake `point + point`). Reach the inner glam
//! value via [`From`]/[`Into`] or the `into_inner` / `from_glam` accessors.

use core::ops::{Add, Div, Mul, Neg, Sub};
use glam::{EulerRot, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

// ---------------------------------------------------------------------------
// Vectors — free directions / displacements (full vector-space algebra).
// ---------------------------------------------------------------------------

macro_rules! impl_vector {
    ($Name:ident, $Inner:ty, $N:literal) => {
        #[doc = concat!("A ", stringify!($N), "-D free vector (direction / displacement).")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(transparent)]
        pub struct $Name($Inner);

        impl $Name {
            /// The zero vector.
            pub const ZERO: Self = Self(<$Inner>::ZERO);

            /// Builds from a column-major array.
            #[inline]
            pub fn from_array(a: [f32; $N]) -> Self {
                Self(<$Inner>::from_array(a))
            }
            /// Returns the components as an array.
            #[inline]
            pub fn to_array(self) -> [f32; $N] {
                self.0.to_array()
            }
            /// Wraps a glam vector.
            #[inline]
            pub fn from_glam(v: $Inner) -> Self {
                Self(v)
            }
            /// Unwraps to the backing glam vector.
            #[inline]
            pub fn into_inner(self) -> $Inner {
                self.0
            }

            /// Dot product.
            #[inline]
            pub fn dot(self, rhs: Self) -> f32 {
                self.0.dot(rhs.0)
            }
            /// Euclidean length.
            #[inline]
            pub fn length(self) -> f32 {
                self.0.length()
            }
            /// Squared length (no `sqrt`).
            #[inline]
            pub fn length_squared(self) -> f32 {
                self.0.length_squared()
            }
            /// Normalizes; **returns NaN** for a ~zero-length vector (glam
            /// semantics). Prefer [`Self::normalize_or_zero`] /
            /// [`Self::try_normalize`] on external data.
            #[inline]
            pub fn normalize(self) -> Self {
                Self(self.0.normalize())
            }
            /// Normalizes, returning [`Self::ZERO`] for a ~zero-length vector.
            #[inline]
            pub fn normalize_or_zero(self) -> Self {
                Self(self.0.normalize_or_zero())
            }
            /// Normalizes, returning `None` for a ~zero-length vector.
            #[inline]
            pub fn try_normalize(self) -> Option<Self> {
                self.0.try_normalize().map(Self)
            }
        }

        impl Add for $Name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl Sub for $Name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }
        impl Neg for $Name {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }
        impl Mul<f32> for $Name {
            type Output = Self;
            #[inline]
            fn mul(self, s: f32) -> Self {
                Self(self.0 * s)
            }
        }
        impl Mul<$Name> for f32 {
            type Output = $Name;
            #[inline]
            fn mul(self, v: $Name) -> $Name {
                $Name(self * v.0)
            }
        }
        impl Div<f32> for $Name {
            type Output = Self;
            #[inline]
            fn div(self, s: f32) -> Self {
                Self(self.0 / s)
            }
        }

        impl From<$Inner> for $Name {
            #[inline]
            fn from(v: $Inner) -> Self {
                Self(v)
            }
        }
        impl From<$Name> for $Inner {
            #[inline]
            fn from(v: $Name) -> Self {
                v.0
            }
        }
        impl From<[f32; $N]> for $Name {
            #[inline]
            fn from(a: [f32; $N]) -> Self {
                Self::from_array(a)
            }
        }
        impl From<$Name> for [f32; $N] {
            #[inline]
            fn from(v: $Name) -> Self {
                v.to_array()
            }
        }
    };
}

impl_vector!(Vector2, Vec2, 2);
impl_vector!(Vector3, Vec3, 3);
impl_vector!(Vector4, Vec4, 4);

impl Vector2 {
    /// The `+x` axis.
    pub const X: Self = Self(Vec2::X);
    /// The `+y` axis.
    pub const Y: Self = Self(Vec2::Y);

    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }
    /// The `x` component.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    /// The `y` component.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
}

impl Vector3 {
    /// The `+x` axis.
    pub const X: Self = Self(Vec3::X);
    /// The `+y` axis.
    pub const Y: Self = Self(Vec3::Y);
    /// The `+z` axis.
    pub const Z: Self = Self(Vec3::Z);

    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
    /// The `x` component.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    /// The `y` component.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
    /// The `z` component.
    #[inline]
    pub fn z(self) -> f32 {
        self.0.z
    }
    /// Cross product.
    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self(self.0.cross(rhs.0))
    }
    /// Lifts to the explicit homogeneous form with `w = 0`.
    #[inline]
    pub fn to_homogeneous(self) -> Vector4 {
        Vector4(self.0.extend(0.0))
    }
}

impl Vector4 {
    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }
    /// The `x` component.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    /// The `y` component.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
    /// The `z` component.
    #[inline]
    pub fn z(self) -> f32 {
        self.0.z
    }
    /// The `w` component.
    #[inline]
    pub fn w(self) -> f32 {
        self.0.w
    }
    /// Drops `w`, keeping the `xyz` direction.
    #[inline]
    pub fn truncate(self) -> Vector3 {
        Vector3(self.0.truncate())
    }
}

// ---------------------------------------------------------------------------
// Points — positions in affine space.
// ---------------------------------------------------------------------------

macro_rules! impl_point {
    ($Name:ident, $Inner:ty, $Vec:ident, $N:literal) => {
        #[doc = concat!("A ", stringify!($N), "-D point (a position in affine space).")]
        #[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(transparent)]
        pub struct $Name($Inner);

        impl $Name {
            /// The coordinate origin.
            pub const ORIGIN: Self = Self(<$Inner>::ZERO);

            /// Builds from a column-major array.
            #[inline]
            pub fn from_array(a: [f32; $N]) -> Self {
                Self(<$Inner>::from_array(a))
            }
            /// Returns the coordinates as an array.
            #[inline]
            pub fn to_array(self) -> [f32; $N] {
                self.0.to_array()
            }
            /// Wraps a glam vector as a point.
            #[inline]
            pub fn from_glam(v: $Inner) -> Self {
                Self(v)
            }
            /// Unwraps to the backing glam vector.
            #[inline]
            pub fn into_inner(self) -> $Inner {
                self.0
            }

            /// The displacement `self → other`, i.e. `other - self` as a vector.
            #[inline]
            pub fn distance(self, other: Self) -> f32 {
                self.0.distance(other.0)
            }
            /// Squared distance (no `sqrt`).
            #[inline]
            pub fn distance_squared(self, other: Self) -> f32 {
                self.0.distance_squared(other.0)
            }
            /// The midpoint of the two positions.
            #[inline]
            pub fn midpoint(self, other: Self) -> Self {
                Self((self.0 + other.0) * 0.5)
            }
            /// Linear interpolation: `t = 0 → self`, `t = 1 → other`.
            #[inline]
            pub fn lerp(self, other: Self, t: f32) -> Self {
                Self(self.0.lerp(other.0, t))
            }
        }

        // Affine algebra: point − point → vector, point ± vector → point.
        // Note: `point + point` is intentionally NOT implemented.
        impl Sub for $Name {
            type Output = $Vec;
            #[inline]
            fn sub(self, rhs: Self) -> $Vec {
                $Vec(self.0 - rhs.0)
            }
        }
        impl Add<$Vec> for $Name {
            type Output = Self;
            #[inline]
            fn add(self, rhs: $Vec) -> Self {
                Self(self.0 + rhs.0)
            }
        }
        impl Sub<$Vec> for $Name {
            type Output = Self;
            #[inline]
            fn sub(self, rhs: $Vec) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl From<$Inner> for $Name {
            #[inline]
            fn from(v: $Inner) -> Self {
                Self(v)
            }
        }
        impl From<$Name> for $Inner {
            #[inline]
            fn from(v: $Name) -> Self {
                v.0
            }
        }
        impl From<[f32; $N]> for $Name {
            #[inline]
            fn from(a: [f32; $N]) -> Self {
                Self::from_array(a)
            }
        }
        impl From<$Name> for [f32; $N] {
            #[inline]
            fn from(v: $Name) -> Self {
                v.to_array()
            }
        }
    };
}

impl_point!(Point2, Vec2, Vector2, 2);
impl_point!(Point3, Vec3, Vector3, 3);
impl_point!(Point4, Vec4, Vector4, 4);

impl Point2 {
    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }
    /// The `x` coordinate.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    /// The `y` coordinate.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
}

impl Point3 {
    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
    /// The `x` coordinate.
    #[inline]
    pub fn x(self) -> f32 {
        self.0.x
    }
    /// The `y` coordinate.
    #[inline]
    pub fn y(self) -> f32 {
        self.0.y
    }
    /// The `z` coordinate.
    #[inline]
    pub fn z(self) -> f32 {
        self.0.z
    }
    /// Lifts to the explicit homogeneous form with `w = 1`.
    #[inline]
    pub fn to_homogeneous(self) -> Point4 {
        Point4(self.0.extend(1.0))
    }
    /// Projects a homogeneous point back to Cartesian (perspective divide).
    #[inline]
    pub fn from_homogeneous(h: Point4) -> Self {
        Self(h.0.truncate() / h.0.w)
    }
}

impl Point4 {
    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }
    /// The `w` (homogeneous) coordinate.
    #[inline]
    pub fn w(self) -> f32 {
        self.0.w
    }
}

// ---------------------------------------------------------------------------
// Normal — a covector: transforms by the inverse-transpose (see `Transform`).
// ---------------------------------------------------------------------------

/// A surface normal (a covector). Transformed by the **inverse-transpose** of a
/// transform's linear part, so it stays perpendicular under non-uniform scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Normal3(Vec3);

impl Normal3 {
    /// Builds from components.
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self(Vec3::new(x, y, z))
    }
    /// Builds from an array.
    #[inline]
    pub fn from_array(a: [f32; 3]) -> Self {
        Self(Vec3::from_array(a))
    }
    /// Returns the components as an array.
    #[inline]
    pub fn to_array(self) -> [f32; 3] {
        self.0.to_array()
    }
    /// Wraps a glam vector.
    #[inline]
    pub fn from_glam(v: Vec3) -> Self {
        Self(v)
    }
    /// Unwraps to the backing glam vector.
    #[inline]
    pub fn into_inner(self) -> Vec3 {
        self.0
    }
    /// Dot product with a direction.
    #[inline]
    pub fn dot(self, v: Vector3) -> f32 {
        self.0.dot(v.0)
    }
    /// Normalizes, returning [`Vector3::ZERO`]-equivalent for ~zero length.
    #[inline]
    pub fn normalize_or_zero(self) -> Self {
        Self(self.0.normalize_or_zero())
    }
    /// The corresponding free direction.
    #[inline]
    pub fn to_vector(self) -> Vector3 {
        Vector3(self.0)
    }
}

impl Neg for Normal3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl From<Vec3> for Normal3 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self(v)
    }
}
impl From<Normal3> for Vec3 {
    #[inline]
    fn from(n: Normal3) -> Self {
        n.0
    }
}

// ---------------------------------------------------------------------------
// Rotation — a *unit* quaternion (the orientation "element").
// ---------------------------------------------------------------------------

/// A 3-D rotation, backed by a **unit** [`glam::Quat`]. The unit invariant means
/// a `Rotation` always denotes a valid rotation (drift can't silently turn it
/// into a skew/scale). This — not a raw quaternion — is what the render/camera
/// path needs; a general non-unit `Quaternion` is intentionally out of v1.
///
/// `Default` is [`Rotation::IDENTITY`] (glam's `Quat::default`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(transparent)]
pub struct Rotation(Quat);

impl Rotation {
    /// The identity (no-op) rotation.
    pub const IDENTITY: Self = Self(Quat::IDENTITY);

    /// Rotation of `rad` radians about a (non-zero) `axis`.
    #[inline]
    pub fn from_axis_angle(axis: Vector3, rad: f32) -> Self {
        Self(Quat::from_axis_angle(axis.0.normalize(), rad))
    }
    /// Rotation about the `+x` axis.
    #[inline]
    pub fn from_rotation_x(rad: f32) -> Self {
        Self(Quat::from_rotation_x(rad))
    }
    /// Rotation about the `+y` axis.
    #[inline]
    pub fn from_rotation_y(rad: f32) -> Self {
        Self(Quat::from_rotation_y(rad))
    }
    /// Rotation about the `+z` axis (the 2D-affine path).
    #[inline]
    pub fn from_rotation_z(rad: f32) -> Self {
        Self(Quat::from_rotation_z(rad))
    }
    /// Rotation from Euler angles in the given `order`.
    #[inline]
    pub fn from_euler(order: EulerRot, a: f32, b: f32, c: f32) -> Self {
        Self(Quat::from_euler(order, a, b, c))
    }
    /// Wraps a quaternion, **normalizing** it to enforce the unit invariant.
    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self(q.normalize())
    }
    /// The backing unit quaternion.
    #[inline]
    pub fn into_inner(self) -> Quat {
        self.0
    }

    /// Shortest-arc spherical interpolation between two rotations.
    #[inline]
    pub fn slerp(self, rhs: Self, t: f32) -> Self {
        Self(self.0.slerp(rhs.0, t))
    }
    /// The inverse rotation (the conjugate, for a unit quaternion).
    #[inline]
    pub fn inverse(self) -> Self {
        Self(self.0.conjugate())
    }
    /// Rotates a free direction.
    #[inline]
    pub fn rotate(self, v: Vector3) -> Vector3 {
        Vector3(self.0 * v.0)
    }
    /// Rotates a point about the origin.
    #[inline]
    pub fn rotate_point(self, p: Point3) -> Point3 {
        Point3(self.0 * p.0)
    }
    /// As a 4×4 rotation matrix.
    #[inline]
    pub fn to_matrix(self) -> Matrix4 {
        Matrix4(Mat4::from_quat(self.0))
    }
    /// As a 3×3 rotation matrix.
    #[inline]
    pub fn to_mat3(self) -> Matrix3 {
        Matrix3(Mat3::from_quat(self.0))
    }
}

impl Mul for Rotation {
    type Output = Self;
    /// Composes rotations; the result stays unit.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl Mul<Vector3> for Rotation {
    type Output = Vector3;
    #[inline]
    fn mul(self, v: Vector3) -> Vector3 {
        self.rotate(v)
    }
}
impl From<Rotation> for Matrix4 {
    #[inline]
    fn from(r: Rotation) -> Self {
        r.to_matrix()
    }
}

// ---------------------------------------------------------------------------
// Matrices — thin newtypes (decision 13). `Default` is IDENTITY (glam).
// ---------------------------------------------------------------------------

/// A column-major 3×3 matrix (the linear part; used for normals).
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Matrix3(Mat3);

impl Default for Matrix3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix3 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self(Mat3::IDENTITY);

    /// Builds from a column-major array of 9 floats.
    #[inline]
    pub fn from_cols_array(a: &[f32; 9]) -> Self {
        Self(Mat3::from_cols_array(a))
    }
    /// Returns the column-major array of 9 floats.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 9] {
        self.0.to_cols_array()
    }
    /// The upper-left 3×3 of a 4×4 matrix.
    #[inline]
    pub fn from_mat4(m: Matrix4) -> Self {
        Self(Mat3::from_mat4(m.0))
    }
    /// Matrix inverse (garbage if singular; see [`Self::try_inverse`]).
    #[inline]
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
    /// Fallible inverse: `None` if (near-)singular.
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        if self.0.determinant().abs() <= f32::EPSILON {
            None
        } else {
            Some(Self(self.0.inverse()))
        }
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self(self.0.transpose())
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.0.determinant()
    }
    /// Wraps a glam matrix.
    #[inline]
    pub fn from_glam(m: Mat3) -> Self {
        Self(m)
    }
    /// Unwraps to the backing glam matrix.
    #[inline]
    pub fn into_inner(self) -> Mat3 {
        self.0
    }
}

impl Mul for Matrix3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

/// A column-major 4×4 matrix — the canonical transform storage and the
/// byte-identical source of the GPU `Uniform` (`to_cols_array` = `[f32; 16]`).
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Matrix4(Mat4);

impl Default for Matrix4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Matrix4 {
    /// The identity matrix.
    pub const IDENTITY: Self = Self(Mat4::IDENTITY);

    /// Builds from a column-major array of 16 floats (protocol / WGSL interop).
    #[inline]
    pub fn from_cols_array(a: &[f32; 16]) -> Self {
        Self(Mat4::from_cols_array(a))
    }
    /// Returns the column-major array of 16 floats (the GPU upload form).
    #[inline]
    pub fn to_cols_array(self) -> [f32; 16] {
        self.0.to_cols_array()
    }
    /// A pure scale.
    #[inline]
    pub fn from_scale(v: Vector3) -> Self {
        Self(Mat4::from_scale(v.0))
    }
    /// A pure translation.
    #[inline]
    pub fn from_translation(v: Vector3) -> Self {
        Self(Mat4::from_translation(v.0))
    }
    /// A pure rotation.
    #[inline]
    pub fn from_rotation(r: Rotation) -> Self {
        Self(Mat4::from_quat(r.0))
    }
    /// Lifts the 3×3 linear part into a 4×4 (zero translation).
    #[inline]
    pub fn from_mat3(m: Matrix3) -> Self {
        Self(Mat4::from_mat3(m.0))
    }
    /// The upper-left 3×3 linear part.
    #[inline]
    pub fn to_mat3(self) -> Matrix3 {
        Matrix3(Mat3::from_mat4(self.0))
    }
    /// Matrix inverse (garbage if singular; see [`Self::try_inverse`]).
    #[inline]
    pub fn inverse(self) -> Self {
        Self(self.0.inverse())
    }
    /// Fallible inverse: `None` if (near-)singular.
    #[inline]
    pub fn try_inverse(self) -> Option<Self> {
        if self.0.determinant().abs() <= f32::EPSILON {
            None
        } else {
            Some(Self(self.0.inverse()))
        }
    }
    /// Transpose.
    #[inline]
    pub fn transpose(self) -> Self {
        Self(self.0.transpose())
    }
    /// Determinant.
    #[inline]
    pub fn determinant(self) -> f32 {
        self.0.determinant()
    }
    /// Wraps a glam matrix.
    #[inline]
    pub fn from_glam(m: Mat4) -> Self {
        Self(m)
    }
    /// Unwraps to the backing glam matrix.
    #[inline]
    pub fn into_inner(self) -> Mat4 {
        self.0
    }
}

impl Mul for Matrix4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl From<Mat4> for Matrix4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        Self(m)
    }
}
impl From<Matrix4> for Mat4 {
    #[inline]
    fn from(m: Matrix4) -> Self {
        m.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::EPSILON;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    /// A finite, well-scaled `f32` — glam's default gen yields NaN/inf which
    /// breaks the algebra laws we assert.
    fn finite() -> impl Strategy<Value = f32> {
        -1000.0f32..1000.0
    }
    fn vec3() -> impl Strategy<Value = Vector3> {
        (finite(), finite(), finite()).prop_map(|(x, y, z)| Vector3::new(x, y, z))
    }
    fn point3() -> impl Strategy<Value = Point3> {
        (finite(), finite(), finite()).prop_map(|(x, y, z)| Point3::new(x, y, z))
    }

    #[test]
    fn newtypes_are_transparent_over_glam() {
        assert_eq!(size_of::<Vector3>(), size_of::<Vec3>());
        assert_eq!(size_of::<Point4>(), size_of::<Vec4>());
        assert_eq!(size_of::<Matrix4>(), 64);
        assert_eq!(size_of::<Matrix3>(), 36);
    }

    #[test]
    fn defaults_are_zero_and_identity() {
        assert_eq!(Vector3::default(), Vector3::ZERO);
        assert_eq!(Point3::default(), Point3::ORIGIN);
        assert_eq!(Matrix4::default(), Matrix4::IDENTITY);
        assert_eq!(Matrix3::default(), Matrix3::IDENTITY);
        assert_eq!(Rotation::default(), Rotation::IDENTITY);
    }

    proptest! {
        #[test]
        fn point_minus_point_is_displacement(p in point3(), q in point3()) {
            // (q - p) added back to p recovers q, up to ~ulp of the larger
            // operand magnitude (the displacement can dwarf |q|).
            let v = q - p;
            let scale = p.into_inner().length().max(q.into_inner().length());
            let tol = 1e-4 * (1.0 + scale);
            assert_abs_diff_eq!((p + v).into_inner(), q.into_inner(), epsilon = tol);
        }

        #[test]
        fn point_plus_then_minus_vector_is_identity(p in point3(), v in vec3()) {
            // The intermediate `p + v` rounds to ~ulp(|v|), so the recovered
            // point can only match `p` to that absolute scale.
            let tol = 1e-4 * (1.0 + v.length());
            assert_abs_diff_eq!((p + v - v).into_inner(), p.into_inner(), epsilon = tol);
        }

        #[test]
        fn vector_add_is_commutative(a in vec3(), b in vec3()) {
            assert_abs_diff_eq!((a + b).into_inner(), (b + a).into_inner(), epsilon = EPSILON);
        }

        #[test]
        fn scalar_mul_commutes(v in vec3(), s in finite()) {
            assert_abs_diff_eq!((v * s).into_inner(), (s * v).into_inner(), epsilon = EPSILON);
        }

        #[test]
        fn cross_is_perpendicular(a in vec3(), b in vec3()) {
            let c = a.cross(b);
            prop_assert!(c.dot(a).abs() <= 1e-2 + 1e-3 * a.length() * b.length());
            prop_assert!(c.dot(b).abs() <= 1e-2 + 1e-3 * a.length() * b.length());
        }

        #[test]
        fn array_round_trip(x in finite(), y in finite(), z in finite()) {
            let v = Vector3::new(x, y, z);
            prop_assert_eq!(Vector3::from_array(v.to_array()), v);
            let arr: [f32; 3] = v.into();
            prop_assert_eq!(Vector3::from(arr), v);
        }

        #[test]
        fn homogeneous_round_trip(p in point3()) {
            let back = Point3::from_homogeneous(p.to_homogeneous());
            assert_abs_diff_eq!(back.into_inner(), p.into_inner(), epsilon = EPSILON);
        }
    }

    #[test]
    fn normalize_edge_cases_are_total() {
        assert_eq!(Vector3::ZERO.normalize_or_zero(), Vector3::ZERO);
        assert_eq!(Vector3::ZERO.try_normalize(), None);
        assert!((Vector3::new(3.0, 0.0, 4.0).normalize().length() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn matrix_inverse_is_fallible_on_singular() {
        assert_eq!(Matrix4::from_scale(Vector3::ZERO).try_inverse(), None);
        assert_eq!(Matrix3::from_glam(Mat3::ZERO).try_inverse(), None);
        let m = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let inv = m.try_inverse().expect("invertible");
        assert_abs_diff_eq!((m * inv).into_inner(), Mat4::IDENTITY, epsilon = EPSILON);
    }

    #[test]
    fn rotation_is_unit_and_composes() {
        let r = Rotation::from_axis_angle(Vector3::new(0.0, 0.0, 2.0), 0.9);
        assert!((r.into_inner().length() - 1.0).abs() < EPSILON);
        // A rotation preserves vector length.
        let v = Vector3::new(1.0, 2.0, 3.0);
        assert!((r.rotate(v).length() - v.length()).abs() < 1e-4);
        // Compose with inverse yields identity rotation.
        let id = r * r.inverse();
        assert_abs_diff_eq!(id.rotate(v).into_inner(), v.into_inner(), epsilon = 1e-4);
    }

    #[test]
    fn from_quat_normalizes() {
        let r = Rotation::from_quat(Quat::from_xyzw(0.0, 0.0, 0.0, 2.0));
        assert!((r.into_inner().length() - 1.0).abs() < EPSILON);
    }
}
