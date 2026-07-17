//! [`ToWgsl`] — converts a math type into its **std140/WGSL-uniform** byte
//! layout as a plain [`bytemuck::Pod`] POD, ready for a uniform buffer upload.
//!
//! WGSL uniform-buffer rules force padding the CPU types don't have:
//! - a `vec3<f32>` occupies **16** bytes (aligned/padded to a `vec4`), and
//! - each column of a `mat3x3<f32>` is a padded `vec3`, so the matrix is
//!   `3 × 16 = 48` bytes laid out as `[[f32; 4]; 3]`.
//!
//! [`Matrix4`] needs no padding: its `[f32; 16]` is already the `mat4x4<f32>`
//! layout, so `to_wgsl` there is byte-identical to `to_cols_array` — that is
//! what the existing render `Uniform` uploads.

use super::linalg::{Matrix3, Matrix4, Normal3, Point2, Point3, Point4, Vector2, Vector3, Vector4};

/// Converts a math value into its WGSL-uniform (std140) POD representation.
pub trait ToWgsl {
    /// The padded, uploadable POD layout.
    type Wgsl: bytemuck::Pod;
    /// Produces the padded layout.
    fn to_wgsl(self) -> Self::Wgsl;
}

impl ToWgsl for Matrix4 {
    type Wgsl = [f32; 16];
    #[inline]
    fn to_wgsl(self) -> [f32; 16] {
        self.to_cols_array()
    }
}

impl ToWgsl for Matrix3 {
    /// Three columns, each a `vec3` padded to 16 bytes.
    type Wgsl = [[f32; 4]; 3];
    #[inline]
    fn to_wgsl(self) -> [[f32; 4]; 3] {
        let c = self.to_cols_array(); // column-major [c0(3), c1(3), c2(3)]
        [
            [c[0], c[1], c[2], 0.0],
            [c[3], c[4], c[5], 0.0],
            [c[6], c[7], c[8], 0.0],
        ]
    }
}

impl ToWgsl for Vector2 {
    type Wgsl = [f32; 2];
    #[inline]
    fn to_wgsl(self) -> [f32; 2] {
        self.to_array()
    }
}

impl ToWgsl for Point2 {
    type Wgsl = [f32; 2];
    #[inline]
    fn to_wgsl(self) -> [f32; 2] {
        self.to_array()
    }
}

impl ToWgsl for Vector3 {
    /// `vec3` padded to 16 bytes with `w = 0` (a direction).
    type Wgsl = [f32; 4];
    #[inline]
    fn to_wgsl(self) -> [f32; 4] {
        let [x, y, z] = self.to_array();
        [x, y, z, 0.0]
    }
}

impl ToWgsl for Normal3 {
    /// `vec3` padded to 16 bytes with `w = 0` (a direction).
    type Wgsl = [f32; 4];
    #[inline]
    fn to_wgsl(self) -> [f32; 4] {
        let [x, y, z] = self.to_array();
        [x, y, z, 0.0]
    }
}

impl ToWgsl for Point3 {
    /// `vec3` padded to 16 bytes with `w = 1` (a position).
    type Wgsl = [f32; 4];
    #[inline]
    fn to_wgsl(self) -> [f32; 4] {
        let [x, y, z] = self.to_array();
        [x, y, z, 1.0]
    }
}

impl ToWgsl for Vector4 {
    type Wgsl = [f32; 4];
    #[inline]
    fn to_wgsl(self) -> [f32; 4] {
        self.to_array()
    }
}

impl ToWgsl for Point4 {
    type Wgsl = [f32; 4];
    #[inline]
    fn to_wgsl(self) -> [f32; 4] {
        self.to_array()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Rotation;

    #[test]
    fn wgsl_layouts_have_std140_sizes() {
        // A mat4 is 64 bytes; a mat3 is three 16-byte-padded columns = 48 bytes.
        assert_eq!(size_of::<<Matrix4 as ToWgsl>::Wgsl>(), 64);
        assert_eq!(size_of::<<Matrix3 as ToWgsl>::Wgsl>(), 48);
        // vec3-derived types pad to a full vec4 (16 bytes).
        assert_eq!(size_of::<<Vector3 as ToWgsl>::Wgsl>(), 16);
        assert_eq!(size_of::<<Point3 as ToWgsl>::Wgsl>(), 16);
        assert_eq!(size_of::<<Normal3 as ToWgsl>::Wgsl>(), 16);
        assert_eq!(size_of::<<Vector2 as ToWgsl>::Wgsl>(), 8);
        assert_eq!(size_of::<<Vector4 as ToWgsl>::Wgsl>(), 16);
    }

    #[test]
    fn matrix4_to_wgsl_matches_cols_array() {
        // The GPU uniform path relies on this being byte-identical.
        let m = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(m.to_wgsl(), m.to_cols_array());
        assert_eq!(
            bytemuck::bytes_of(&m.to_wgsl()),
            bytemuck::cast_slice::<f32, u8>(&m.to_cols_array())
        );
    }

    #[test]
    fn direction_and_position_use_correct_w() {
        assert_eq!(Vector3::new(1.0, 2.0, 3.0).to_wgsl(), [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(Normal3::new(1.0, 2.0, 3.0).to_wgsl(), [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(Point3::new(1.0, 2.0, 3.0).to_wgsl(), [1.0, 2.0, 3.0, 1.0]);
    }

    #[test]
    fn mat3_wgsl_pads_each_column() {
        let m = Rotation::from_rotation_z(0.0).to_mat3(); // identity 3x3
        let w = m.to_wgsl();
        assert_eq!(w[0], [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(w[1], [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(w[2], [0.0, 0.0, 1.0, 0.0]);
    }
}
