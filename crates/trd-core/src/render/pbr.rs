//! GPU packing and smooth-normal derivation for the Disney PBR path.

use super::{
    ImageBasedLighting, Lighting, ToneMapping, Vertex, DEFAULT_LIGHTS, DEFAULT_POINT_LIGHTS,
};
use crate::material::DisneyMaterial;
use crate::math::Vector3;

/// Diagnostic output of one PBR material input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PbrDebugView {
    #[default]
    Shaded,
    Roughness,
    Metallic,
    Normal,
}

impl PbrDebugView {
    fn to_uniform(self) -> f32 {
        match self {
            Self::Shaded => 1.0,
            Self::Roughness => 2.0,
            Self::Metallic => 3.0,
            Self::Normal => 4.0,
        }
    }
}

/// Maximum lights per kind, matching `pbr.wgsl`'s `MAX_LIGHTS` array size.
const MAX_LIGHTS: usize = 4;

pub(crate) struct PbrUniformInputs<'a> {
    pub material: &'a DisneyMaterial,
    pub lighting: Lighting,
    pub ibl: ImageBasedLighting,
    pub tone_mapping: ToneMapping,
    pub debug_view: PbrDebugView,
    pub use_env: bool,
}

/// The **batched** twin of [`PbrUniformInputs`]: the same five inputs, pluralized
/// per mesh, as `SceneRenderer::encode` receives them for a whole frame.
/// `write_pbr` slices one `PbrUniformInputs` out of it per mesh id.
///
/// `lighting` is singular because the light rig is scene-level, not per mesh —
/// a layering wart tracked in #182.
pub(crate) struct PbrBatchInputs<'a> {
    pub materials: &'a [DisneyMaterial],
    pub ibl: &'a [ImageBasedLighting],
    pub tone_mappings: &'a [ToneMapping],
    pub debug_views: &'a [PbrDebugView],
    pub lighting: Lighting,
}

/// GPU byte layout matching `pbr.wgsl`'s `PbrUniform` (std140-compatible: all
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
        inputs: PbrUniformInputs<'_>,
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
            camera_pos: [
                camera_pos[0],
                camera_pos[1],
                camera_pos[2],
                inputs.debug_view.to_uniform(),
            ],
            mat0: [
                inputs.material.metallic,
                inputs.material.subsurface,
                inputs.material.specular,
                inputs.material.roughness,
            ],
            mat1: [
                inputs.material.specular_tint,
                inputs.material.anisotropic,
                inputs.material.sheen,
                inputs.material.sheen_tint,
            ],
            mat2: [
                inputs.material.clearcoat,
                inputs.material.clearcoat_gloss,
                inputs.ibl.intensity,
                inputs.tone_mapping.exposure,
            ],
            mat3: [
                inputs.material.base_color[0],
                inputs.material.base_color[1],
                inputs.material.base_color[2],
                inputs.lighting.ambient,
            ],
            counts: [
                DEFAULT_LIGHTS.len() as f32,
                DEFAULT_POINT_LIGHTS.len() as f32,
                if inputs.use_env { 1.0 } else { 0.0 },
                inputs.lighting.scale,
            ],
            mat4: [
                inputs.tone_mapping.operator.to_uniform(),
                inputs.ibl.rotation,
                if inputs.material.auxiliary.textures.normal {
                    1.0
                } else {
                    0.0
                },
                if inputs.material.auxiliary.textures.metallic_roughness {
                    1.0
                } else {
                    0.0
                },
            ],
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

/// Derives a MikkTSpace-compatible tangent basis approximation from positions
/// and UVs, orthonormalized against the shading normal.
pub(crate) fn compute_tangents(
    vertices: &[Vertex],
    indices: &[u32],
    normals: &[[f32; 3]],
) -> Vec<[f32; 4]> {
    let mut tangents = vec![Vector3::ZERO; vertices.len()];
    let mut bitangents = vec![Vector3::ZERO; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let p0 = Vector3::from_array(vertices[i0].position);
        let p1 = Vector3::from_array(vertices[i1].position);
        let p2 = Vector3::from_array(vertices[i2].position);
        let uv0 = vertices[i0].uv;
        let uv1 = vertices[i1].uv;
        let uv2 = vertices[i2].uv;
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];
        let det = duv1[0] * duv2[1] - duv1[1] * duv2[0];
        if det.abs() <= 1e-12 {
            continue;
        }
        let r = det.recip();
        let tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * r;
        let bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) * r;
        for i in [i0, i1, i2] {
            tangents[i] = tangents[i] + tangent;
            bitangents[i] = bitangents[i] + bitangent;
        }
    }
    normals
        .iter()
        .enumerate()
        .map(|(i, normal)| {
            let n = Vector3::from_array(*normal);
            let mut t = tangents[i] - n * n.dot(tangents[i]);
            if t.length() <= 1e-12 {
                let axis = if n.x().abs() < 0.9 {
                    Vector3::X
                } else {
                    Vector3::Y
                };
                t = n.cross(axis);
            }
            t = t.normalize();
            let handedness = if n.cross(t).dot(bitangents[i]) < 0.0 {
                -1.0
            } else {
                1.0
            };
            let t = t.to_array();
            [t[0], t[1], t[2], handedness]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tonemap;

    fn inputs(material: &DisneyMaterial, use_env: bool) -> PbrUniformInputs<'_> {
        PbrUniformInputs {
            material,
            lighting: Lighting::default(),
            ibl: ImageBasedLighting::default(),
            tone_mapping: ToneMapping::default(),
            debug_view: PbrDebugView::Shaded,
            use_env,
        }
    }

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
        let with = PbrUniform::new([0.0; 16], [0.0; 3], inputs(&m, true));
        let without = PbrUniform::new([0.0; 16], [0.0; 3], inputs(&m, false));
        assert_eq!(with.counts[2], 1.0);
        assert_eq!(without.counts[2], 0.0);
        assert_eq!(with.counts[0], DEFAULT_LIGHTS.len() as f32);
    }

    #[test]
    fn tonemap_packs_into_mat4_x_and_defaults_reinhard() {
        let reinhard = PbrUniform::new(
            [0.0; 16],
            [0.0; 3],
            inputs(&DisneyMaterial::default(), false),
        );
        assert_eq!(reinhard.mat4[0], 0.0);
        let tone_mapping = ToneMapping {
            operator: Tonemap::Aces,
            ..ToneMapping::default()
        };
        let material = DisneyMaterial::default();
        let mut inputs = inputs(&material, false);
        inputs.tone_mapping = tone_mapping;
        let u = PbrUniform::new([0.0; 16], [0.0; 3], inputs);
        assert_eq!(u.mat4[0], 1.0);
        assert_eq!([u.mat4[1], u.mat4[2], u.mat4[3]], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn typed_domains_preserve_legacy_default_uniform_bytes() {
        let uniform = PbrUniform::new(
            [0.0; 16],
            [0.0; 3],
            inputs(&DisneyMaterial::default(), true),
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
