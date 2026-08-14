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

/// The per-**mesh** PBR inputs. Every field describes one object; the light rig
/// and the camera are scene-wide and live in [`PbrSceneUniform`] instead (#182).
pub(crate) struct PbrUniformInputs<'a> {
    pub material: &'a DisneyMaterial,
    pub ibl: ImageBasedLighting,
    pub tone_mapping: ToneMapping,
    pub debug_view: PbrDebugView,
}

/// GPU byte layout matching `pbr.wgsl`'s `PbrSceneUniform` (group 0, binding 0):
/// **224 bytes written once per frame**, holding everything the whole frame
/// shares — the camera terms and the light rig.
///
/// It used to be re-encoded into every per-mesh slot, so an N-object scene wrote
/// N identical copies of the same lights each frame (#182). Splitting the group
/// by *frequency of change* is also what shrinks the per-mesh slot from 304 to
/// 80 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PbrSceneUniform {
    view_proj: [f32; 16],
    /// xyz = camera world position, w = the env gate (1 when a probe is bound).
    camera_pos: [f32; 4],
    /// num_dir_lights, num_point_lights, ambient, light_scale.
    light_params: [f32; 4],
    /// xyz = direction the light travels, w = intensity.
    dir_lights: [[f32; 4]; MAX_LIGHTS],
    /// xyz = world position, w = intensity.
    point_lights: [[f32; 4]; MAX_LIGHTS],
}

impl PbrSceneUniform {
    /// Packs this frame's camera + light rig. `lighting` arrives on the
    /// [`Scene`](crate::Scene) now, so the rig travels *with* the frame and is
    /// written *once* for it.
    pub(crate) fn new(
        view_proj: [f32; 16],
        camera_pos: [f32; 3],
        lighting: Lighting,
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
            camera_pos: [
                camera_pos[0],
                camera_pos[1],
                camera_pos[2],
                if use_env { 1.0 } else { 0.0 },
            ],
            light_params: [
                DEFAULT_LIGHTS.len() as f32,
                DEFAULT_POINT_LIGHTS.len() as f32,
                lighting.ambient,
                lighting.scale,
            ],
            dir_lights,
            point_lights,
        }
    }
}

/// GPU byte layout matching `pbr.wgsl`'s `PbrUniform` (group 0, binding 1): the
/// **80-byte per-mesh slot** a draw selects with a dynamic offset — material,
/// IBL gain, tone map and debug view, and nothing the frame already shares.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PbrUniform {
    /// metallic, subsurface, specular, roughness
    mat0: [f32; 4],
    /// specularTint, anisotropic, sheen, sheenTint
    mat1: [f32; 4],
    /// clearcoat, clearcoatGloss, env_intensity, exposure
    mat2: [f32; 4],
    /// baseColorTint.rgb, debug view
    mat3: [f32; 4],
    /// tonemap mode, ibl rotation, has normal map, has metallic-roughness map
    mat4: [f32; 4],
}

impl PbrUniform {
    /// Composes the typed per-object PBR domains into the shader's slot layout.
    pub(crate) fn new(inputs: PbrUniformInputs<'_>) -> Self {
        Self {
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
                inputs.debug_view.to_uniform(),
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

    fn inputs(material: &DisneyMaterial) -> PbrUniformInputs<'_> {
        PbrUniformInputs {
            material,
            ibl: ImageBasedLighting::default(),
            tone_mapping: ToneMapping::default(),
            debug_view: PbrDebugView::Shaded,
        }
    }

    #[test]
    fn uniforms_split_by_frequency_of_change() {
        // The rig is written once per frame (224 bytes); a mesh slot carries only
        // its own material (80 bytes), down from 304 when both shared one struct.
        assert_eq!(std::mem::size_of::<PbrSceneUniform>(), 224);
        assert_eq!(std::mem::size_of::<PbrUniform>(), 80);
        assert_eq!(std::mem::size_of::<PbrSceneUniform>() % 16, 0);
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
    fn use_env_flag_packs_into_the_scene_camera_w() {
        let lighting = Lighting::default();
        let with = PbrSceneUniform::new([0.0; 16], [0.0; 3], lighting, true);
        let without = PbrSceneUniform::new([0.0; 16], [0.0; 3], lighting, false);
        assert_eq!(with.camera_pos[3], 1.0);
        assert_eq!(without.camera_pos[3], 0.0);
        assert_eq!(with.light_params[0], DEFAULT_LIGHTS.len() as f32);
    }

    #[test]
    fn scene_uniform_carries_the_rig_the_slots_no_longer_repeat() {
        let lighting = Lighting {
            ambient: 0.25,
            scale: 3.0,
        };
        let scene = PbrSceneUniform::new([0.0; 16], [1.0, 2.0, 3.0], lighting, false);
        assert_eq!(scene.light_params[2], 0.25);
        assert_eq!(scene.light_params[3], 3.0);
        assert_eq!([scene.camera_pos[0], scene.camera_pos[1]], [1.0, 2.0]);
        // The per-mesh slot has no light rig left to disagree with it.
        let slot = PbrUniform::new(inputs(&DisneyMaterial::default()));
        assert_eq!(std::mem::size_of_val(&slot), 80);
    }

    #[test]
    fn tonemap_packs_into_mat4_x_and_defaults_reinhard() {
        let reinhard = PbrUniform::new(inputs(&DisneyMaterial::default()));
        assert_eq!(reinhard.mat4[0], 0.0);
        let tone_mapping = ToneMapping {
            operator: Tonemap::Aces,
            ..ToneMapping::default()
        };
        let material = DisneyMaterial::default();
        let mut inputs = inputs(&material);
        inputs.tone_mapping = tone_mapping;
        let u = PbrUniform::new(inputs);
        assert_eq!(u.mat4[0], 1.0);
        assert_eq!([u.mat4[1], u.mat4[2], u.mat4[3]], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn typed_domains_preserve_the_packed_values_across_the_split() {
        // The same numbers the single 304-byte uniform used to carry, now in the
        // two halves: the rig + camera terms once, the material per mesh.
        let scene = PbrSceneUniform::new([0.0; 16], [0.0; 3], Lighting::default(), true);
        let expected_scene = PbrSceneUniform {
            view_proj: [0.0; 16],
            // w = use_env, which used to sit in `counts.z`.
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            // num dir, num point, ambient (was mat3.w), light scale (was counts.w).
            light_params: [3.0, 0.0, 0.12, 2.5],
            dir_lights: [
                [-0.5, -0.85, -0.55, 1.0],
                [0.8, -0.25, 0.35, 0.4],
                [0.25, -0.3, 0.9, 0.55],
                [0.0; 4],
            ],
            point_lights: [[0.0; 4]; MAX_LIGHTS],
        };
        assert_eq!(
            bytemuck::bytes_of(&scene),
            bytemuck::bytes_of(&expected_scene)
        );

        let slot = PbrUniform::new(inputs(&DisneyMaterial::default()));
        let expected_slot = PbrUniform {
            mat0: [0.0, 0.0, 0.5, 0.5],
            mat1: [0.0, 0.0, 0.0, 0.5],
            mat2: [0.0, 1.0, 1.0, 1.2],
            // baseColorTint.rgb + the debug view, which used to sit in camera_pos.w.
            mat3: [1.0, 1.0, 1.0, 1.0],
            mat4: [0.0; 4],
        };
        assert_eq!(
            bytemuck::bytes_of(&slot),
            bytemuck::bytes_of(&expected_slot)
        );
    }
}
