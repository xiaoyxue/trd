//! GPU packing and smooth-normal derivation for the Disney PBR path.

use super::{
    DisneyMaterial, ImageBasedLighting, Lighting, ToneMapping, Vertex, DEFAULT_LIGHTS,
    DEFAULT_POINT_LIGHTS,
};
use crate::math::Vector3;

/// Maximum lights per kind, matching `disney.wgsl`'s `MAX_LIGHTS` array size.
const MAX_LIGHTS: usize = 4;

/// GPU byte layout matching `disney.wgsl`'s `PbrUniform` (std140-compatible: all
/// members are 16-byte-aligned `vec4`/`mat4`). 304 bytes, uploaded per frame
/// (the camera terms change; the material + light rig are constant per render).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PbrUniform {
    view_proj: [f32; 16],
    camera_pos: [f32; 4],
    mat0: [f32; 4],
    mat1: [f32; 4],
    mat2: [f32; 4],
    mat3: [f32; 4],
    counts: [f32; 4],
    mat4: [f32; 4],
    dir_lights: [[f32; 4]; MAX_LIGHTS],
    point_lights: [[f32; 4]; MAX_LIGHTS],
}

impl PbrUniform {
    /// Composes the typed PBR domains into the shader's unchanged byte layout.
    pub(crate) fn new(
        view_proj: [f32; 16],
        camera_pos: [f32; 3],
        material: &DisneyMaterial,
        lighting: Lighting,
        ibl: ImageBasedLighting,
        tone_mapping: ToneMapping,
        use_env: bool,
    ) -> Self {
        let mut dir_lights = [[0.0f32; 4]; MAX_LIGHTS];
        for (packed, light) in dir_lights.iter_mut().zip(DEFAULT_LIGHTS) {
            *packed = light.to_uniform();
        }
        let mut point_lights = [[0.0f32; 4]; MAX_LIGHTS];
        for (packed, light) in point_lights.iter_mut().zip(DEFAULT_POINT_LIGHTS) {
            *packed = light.to_uniform();
        }
        Self {
            view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
            mat0: [
                material.metallic,
                material.subsurface,
                material.specular,
                material.roughness,
            ],
            mat1: [
                material.specular_tint,
                material.anisotropic,
                material.sheen,
                material.sheen_tint,
            ],
            mat2: [
                material.clearcoat,
                material.clearcoat_gloss,
                ibl.intensity,
                tone_mapping.exposure,
            ],
            mat3: [
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
                lighting.ambient,
            ],
            counts: [
                DEFAULT_LIGHTS.len() as f32,
                DEFAULT_POINT_LIGHTS.len() as f32,
                if use_env { 1.0 } else { 0.0 },
                lighting.scale,
            ],
            mat4: [tone_mapping.operator.to_uniform(), 0.0, 0.0, 0.0],
            dir_lights,
            point_lights,
        }
    }
}

/// Computes area-weighted smooth per-vertex normals from an indexed triangle
/// mesh. The trd assets carry no `vn`, so the Disney path derives shading normals
/// here: each triangle's un-normalized cross product (whose length is ∝ twice its
/// area) is accumulated onto its three vertices, then each vertex normal is
/// normalized. Degenerate (zero-length) accumulations fall back to `+Z`.
pub(crate) fn compute_smooth_normals(vertices: &[Vertex], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![Vector3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vector3::from_array(vertices[i0].position);
        let p1 = Vector3::from_array(vertices[i1].position);
        let p2 = Vector3::from_array(vertices[i2].position);
        // Length ∝ 2× triangle area, so larger faces weight the shared normal
        // more — the standard area-weighted smoothing.
        let face = (p1 - p0).cross(p2 - p0);
        normals[i0] = normals[i0] + face;
        normals[i1] = normals[i1] + face;
        normals[i2] = normals[i2] + face;
    }
    normals
        .into_iter()
        .map(|n| {
            let len = n.length();
            if len > 1e-12 {
                (n / len).to_array()
            } else {
                [0.0, 0.0, 1.0]
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tonemap;

    #[test]
    fn pbr_uniform_is_304_bytes() {
        assert_eq!(std::mem::size_of::<PbrUniform>(), 304);
        assert_eq!(std::mem::size_of::<PbrUniform>() % 16, 0);
    }

    #[test]
    fn smooth_normals_of_xy_quad_point_up_z() {
        // Two triangles of a unit quad in the z=0 plane, CCW → +Z normals.
        let v = |x: f32, y: f32| Vertex {
            position: [x, y, 0.0],
            color: [0.0; 3],
            uv: [0.0; 2],
        };
        let vertices = vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)];
        let indices = vec![0, 1, 2, 0, 2, 3];
        let normals = compute_smooth_normals(&vertices, &indices);
        for n in normals {
            assert!((n[0]).abs() < 1e-6);
            assert!((n[1]).abs() < 1e-6);
            assert!((n[2] - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn use_env_flag_packs_into_counts() {
        let m = DisneyMaterial::default();
        let lighting = Lighting::default();
        let ibl = ImageBasedLighting::default();
        let tone_mapping = ToneMapping::default();
        let with = PbrUniform::new([0.0; 16], [0.0; 3], &m, lighting, ibl, tone_mapping, true);
        let without = PbrUniform::new([0.0; 16], [0.0; 3], &m, lighting, ibl, tone_mapping, false);
        assert_eq!(with.counts[2], 1.0);
        assert_eq!(without.counts[2], 0.0);
        assert_eq!(with.counts[0], DEFAULT_LIGHTS.len() as f32);
    }

    #[test]
    fn tonemap_packs_into_mat4_x_and_defaults_reinhard() {
        let reinhard = PbrUniform::new(
            [0.0; 16],
            [0.0; 3],
            &DisneyMaterial::default(),
            Lighting::default(),
            ImageBasedLighting::default(),
            ToneMapping::default(),
            false,
        );
        assert_eq!(reinhard.mat4[0], 0.0);
        let tone_mapping = ToneMapping {
            operator: Tonemap::Aces,
            ..ToneMapping::default()
        };
        let u = PbrUniform::new(
            [0.0; 16],
            [0.0; 3],
            &DisneyMaterial::default(),
            Lighting::default(),
            ImageBasedLighting::default(),
            tone_mapping,
            false,
        );
        assert_eq!(u.mat4[0], 1.0);
        assert_eq!([u.mat4[1], u.mat4[2], u.mat4[3]], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn typed_domains_preserve_legacy_default_uniform_bytes() {
        let uniform = PbrUniform::new(
            [0.0; 16],
            [0.0; 3],
            &DisneyMaterial::default(),
            Lighting::default(),
            ImageBasedLighting::default(),
            ToneMapping::default(),
            true,
        );
        let expected = PbrUniform {
            view_proj: [0.0; 16],
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            mat0: [0.0, 0.0, 0.5, 0.5],
            mat1: [0.0, 0.0, 0.0, 0.5],
            mat2: [0.0, 1.0, 1.0, 1.2],
            mat3: [1.0, 1.0, 1.0, 0.12],
            counts: [3.0, 0.0, 1.0, 2.5],
            mat4: [0.0; 4],
            dir_lights: [
                [-0.5, -0.85, -0.55, 1.0],
                [0.8, -0.25, 0.35, 0.4],
                [0.25, -0.3, 0.9, 0.55],
                [0.0; 4],
            ],
            point_lights: [[0.0; 4]; MAX_LIGHTS],
        };
        assert_eq!(bytemuck::bytes_of(&uniform), bytemuck::bytes_of(&expected));
    }
}
