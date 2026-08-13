//! glTF material import into trd's typed Disney model.

use std::collections::BTreeMap;

use crate::{
    AlphaMode, Auxiliary, DisneyMaterial, ImageTexture, MaterialTextures, Mesh, MeshShading,
    TextureError, Vertex,
};
use thiserror::Error;

const DEFAULT_IOR: f32 = 1.5;
const MAX_TEXTURE_DIM: u32 = 2048;

/// One renderable glTF primitive and its Disney PBR resources.
#[derive(Debug, Clone, PartialEq)]
pub struct GltfAsset {
    pub mesh: Mesh,
    pub material: DisneyMaterial,
    pub base_color_texture: Option<ImageTexture>,
    pub metallic_roughness_texture: Option<ImageTexture>,
    pub normal_texture: Option<ImageTexture>,
}

#[derive(Debug, Error)]
pub enum GltfImportError {
    #[error("invalid glTF: {0}")]
    Gltf(#[from] gltf_rs::Error),
    #[error("glTF asset has no mesh primitive")]
    MissingPrimitive,
    #[error("glTF import currently requires one mesh with one primitive")]
    MultiplePrimitives,
    #[error("glTF primitive mode {0:?} is unsupported (expected triangles)")]
    PrimitiveMode(gltf_rs::mesh::Mode),
    #[error("glTF primitive has no POSITION attribute")]
    MissingPositions,
    #[error("glTF buffer {0} is external; GLB BIN data is required")]
    ExternalBuffer(usize),
    #[error("GLB has no BIN chunk")]
    MissingBlob,
    #[error("glTF base-color image is external; an embedded GLB image is required")]
    ExternalImage,
    #[error("glTF embedded image buffer view is outside the GLB BIN chunk")]
    ImageViewOutOfBounds,
    #[error("failed to decode glTF base-color image: {0}")]
    Image(#[from] image::ImageError),
    #[error("invalid decoded glTF base-color texture: {0}")]
    Texture(#[from] TextureError),
    #[error("glTF extension `{extension}` field `{field}` must be {expected}")]
    MalformedExtension {
        extension: &'static str,
        field: &'static str,
        expected: &'static str,
    },
}

/// Parses one binary glTF primitive, its Disney material, and embedded albedo.
///
/// This is the first runtime GLB slice. It deliberately rejects multi-primitive
/// assets rather than separating their pieces into unrelated GUI objects.
pub fn import_glb(bytes: &[u8]) -> Result<GltfAsset, GltfImportError> {
    let gltf = gltf_rs::Gltf::from_slice(bytes)?;
    let blob = gltf.blob.as_deref().ok_or(GltfImportError::MissingBlob)?;
    let mut primitives = gltf.document.meshes().flat_map(|mesh| mesh.primitives());
    let primitive = primitives.next().ok_or(GltfImportError::MissingPrimitive)?;
    if primitives.next().is_some() {
        return Err(GltfImportError::MultiplePrimitives);
    }
    if primitive.mode() != gltf_rs::mesh::Mode::Triangles {
        return Err(GltfImportError::PrimitiveMode(primitive.mode()));
    }

    let reader = primitive.reader(|buffer| match buffer.source() {
        gltf_rs::buffer::Source::Bin => Some(blob),
        gltf_rs::buffer::Source::Uri(_) => None,
    });
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(GltfImportError::MissingPositions)?
        .collect();
    let tex_coords: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0; 2]; positions.len()]);
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(Iterator::collect);
    let tangents: Vec<[f32; 4]> = reader
        .read_tangents()
        .map(Iterator::collect)
        .unwrap_or_default();
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|indices| indices.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    let vertices: Vec<Vertex> = positions
        .into_iter()
        .zip(tex_coords)
        .map(|(position, uv)| Vertex {
            position,
            color: [1.0; 3],
            uv,
        })
        .collect();

    let source_material = primitive.material();
    let material = import_material(source_material.clone())?;
    let pbr = source_material.pbr_metallic_roughness();
    let base_color_texture = pbr
        .base_color_texture()
        .map(|info| import_embedded_texture(info.texture().source(), blob))
        .transpose()?;
    let metallic_roughness_texture = pbr
        .metallic_roughness_texture()
        .map(|info| import_embedded_texture(info.texture().source(), blob))
        .transpose()?;
    let normal_texture = source_material
        .normal_texture()
        .map(|info| import_embedded_texture(info.texture().source(), blob))
        .transpose()?;
    Ok(GltfAsset {
        mesh: Mesh {
            shading: normals
                .filter(|normals| normals.len() == vertices.len())
                .map(|normals| MeshShading { normals, tangents }),
            vertices,
            indices,
        },
        material,
        base_color_texture,
        metallic_roughness_texture,
        normal_texture,
    })
}

/// Parses every explicit glTF material without loading geometry or image bytes.
pub fn import_gltf_materials(bytes: &[u8]) -> Result<Vec<DisneyMaterial>, GltfImportError> {
    let gltf = gltf_rs::Gltf::from_slice(bytes)?;
    gltf.document.materials().map(import_material).collect()
}

fn import_embedded_texture(
    image: gltf_rs::Image<'_>,
    blob: &[u8],
) -> Result<ImageTexture, GltfImportError> {
    let gltf_rs::image::Source::View { view, .. } = image.source() else {
        return Err(GltfImportError::ExternalImage);
    };
    let start = view.offset();
    let end = start
        .checked_add(view.length())
        .ok_or(GltfImportError::ImageViewOutOfBounds)?;
    let encoded = blob
        .get(start..end)
        .ok_or(GltfImportError::ImageViewOutOfBounds)?;
    let decoded = image::load_from_memory(encoded)?;
    let decoded = if decoded.width() > MAX_TEXTURE_DIM || decoded.height() > MAX_TEXTURE_DIM {
        decoded.resize(
            MAX_TEXTURE_DIM,
            MAX_TEXTURE_DIM,
            image::imageops::FilterType::Triangle,
        )
    } else {
        decoded
    };
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(ImageTexture::from_rgba(width, height, rgba.into_raw())?)
}

fn import_material(material: gltf_rs::Material<'_>) -> Result<DisneyMaterial, GltfImportError> {
    let pbr = material.pbr_metallic_roughness();
    let base_color_factor = pbr.base_color_factor();
    let ior = material.ior().unwrap_or(DEFAULT_IOR);
    let specular_factor = material
        .specular()
        .map(|specular| specular.specular_factor())
        .unwrap_or(1.0);
    let dielectric_f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);

    let clearcoat =
        extension_number(&material, "KHR_materials_clearcoat", "clearcoatFactor")?.unwrap_or(0.0);
    let clearcoat_roughness = extension_number(
        &material,
        "KHR_materials_clearcoat",
        "clearcoatRoughnessFactor",
    )?
    .unwrap_or(0.0);
    let anisotropic =
        extension_number(&material, "KHR_materials_anisotropy", "anisotropyStrength")?
            .unwrap_or(0.0);
    let sheen_color =
        extension_vec3(&material, "KHR_materials_sheen", "sheenColorFactor")?.unwrap_or([0.0; 3]);

    let mut sources = BTreeMap::new();
    sources.insert(
        "base_color".into(),
        "gltf:pbrMetallicRoughness.baseColorFactor".into(),
    );
    sources.insert(
        "metallic".into(),
        "gltf:pbrMetallicRoughness.metallicFactor".into(),
    );
    sources.insert(
        "roughness".into(),
        "gltf:pbrMetallicRoughness.roughnessFactor".into(),
    );
    sources.insert("subsurface".into(), "default:0".into());
    sources.insert(
        "specular".into(),
        "derived:gltf IOR F0 / 0.08 * KHR_materials_specular.specularFactor".into(),
    );
    sources.insert("specular_tint".into(), "default:0".into());
    sources.insert(
        "anisotropic".into(),
        source_or_default(
            material
                .extension_value("KHR_materials_anisotropy")
                .is_some(),
            "gltf:KHR_materials_anisotropy.anisotropyStrength",
            "default:0",
        ),
    );
    sources.insert(
        "sheen".into(),
        source_or_default(
            material.extension_value("KHR_materials_sheen").is_some(),
            "gltf:KHR_materials_sheen.sheenColorFactor(max component)",
            "default:0",
        ),
    );
    sources.insert("sheen_tint".into(), "default:0.5".into());
    sources.insert(
        "clearcoat".into(),
        source_or_default(
            material
                .extension_value("KHR_materials_clearcoat")
                .is_some(),
            "gltf:KHR_materials_clearcoat.clearcoatFactor",
            "default:0",
        ),
    );
    sources.insert(
        "clearcoat_gloss".into(),
        source_or_default(
            material
                .extension_value("KHR_materials_clearcoat")
                .is_some(),
            "derived:1 - KHR_materials_clearcoat.clearcoatRoughnessFactor",
            "default:1",
        ),
    );

    let transmission = material
        .transmission()
        .map(|value| value.transmission_factor())
        .unwrap_or(0.0);
    let auxiliary = Auxiliary {
        opacity: base_color_factor[3],
        alpha_mode: match material.alpha_mode() {
            gltf_rs::material::AlphaMode::Opaque => AlphaMode::Opaque,
            gltf_rs::material::AlphaMode::Mask => AlphaMode::Mask,
            gltf_rs::material::AlphaMode::Blend => AlphaMode::Blend,
        },
        alpha_cutoff: material.alpha_cutoff(),
        double_sided: material.double_sided(),
        emissive: material.emissive_factor(),
        emissive_strength: material.emissive_strength().unwrap_or(1.0),
        ior,
        transmission,
        textures: MaterialTextures {
            base_color: pbr.base_color_texture().is_some(),
            metallic_roughness: pbr.metallic_roughness_texture().is_some(),
            normal: material.normal_texture().is_some(),
            occlusion: material.occlusion_texture().is_some(),
            emissive: material.emissive_texture().is_some(),
        },
    };

    Ok(DisneyMaterial {
        name: material.name().map(str::to_owned),
        base_color: [
            base_color_factor[0],
            base_color_factor[1],
            base_color_factor[2],
        ],
        metallic: pbr.metallic_factor(),
        subsurface: 0.0,
        specular: (dielectric_f0 / 0.08 * specular_factor).clamp(0.0, 1.0),
        roughness: pbr.roughness_factor(),
        specular_tint: 0.0,
        anisotropic,
        sheen: sheen_color.into_iter().fold(0.0, f32::max),
        sheen_tint: 0.5,
        clearcoat,
        clearcoat_gloss: 1.0 - clearcoat_roughness,
        auxiliary,
        sources,
    })
}

fn extension_number(
    material: &gltf_rs::Material<'_>,
    extension: &'static str,
    field: &'static str,
) -> Result<Option<f32>, GltfImportError> {
    let Some(value) = material
        .extension_value(extension)
        .and_then(|extension| extension.get(field))
    else {
        return Ok(None);
    };
    value
        .as_f64()
        .map(|value| Some(value as f32))
        .ok_or(GltfImportError::MalformedExtension {
            extension,
            field,
            expected: "a number",
        })
}

fn extension_vec3(
    material: &gltf_rs::Material<'_>,
    extension: &'static str,
    field: &'static str,
) -> Result<Option<[f32; 3]>, GltfImportError> {
    let Some(value) = material
        .extension_value(extension)
        .and_then(|extension| extension.get(field))
    else {
        return Ok(None);
    };
    let Some(values) = value.as_array() else {
        return Err(GltfImportError::MalformedExtension {
            extension,
            field,
            expected: "an array of three numbers",
        });
    };
    if values.len() != 3 {
        return Err(GltfImportError::MalformedExtension {
            extension,
            field,
            expected: "an array of three numbers",
        });
    }
    let mut parsed = [0.0; 3];
    for (slot, value) in parsed.iter_mut().zip(values) {
        *slot = value.as_f64().map(|value| value as f32).ok_or(
            GltfImportError::MalformedExtension {
                extension,
                field,
                expected: "an array of three numbers",
            },
        )?;
    }
    Ok(Some(parsed))
}

fn source_or_default(present: bool, present_source: &str, default_source: &str) -> String {
    if present {
        present_source.to_owned()
    } else {
        default_source.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_triangle_glb() -> Vec<u8> {
        let mut bin = Vec::new();
        for value in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0] {
            bin.extend_from_slice(&value.to_le_bytes());
        }
        for index in [0u16, 1, 2] {
            bin.extend_from_slice(&index.to_le_bytes());
        }
        let declared_bin_len = bin.len();
        while bin.len() % 4 != 0 {
            bin.push(0);
        }

        let mut json = format!(
            r#"{{
              "asset":{{"version":"2.0"}},
              "buffers":[{{"byteLength":{declared_bin_len}}}],
              "bufferViews":[
                {{"buffer":0,"byteOffset":0,"byteLength":36}},
                {{"buffer":0,"byteOffset":36,"byteLength":24}},
                {{"buffer":0,"byteOffset":60,"byteLength":6}}
              ],
              "accessors":[
                {{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}},
                {{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}},
                {{"bufferView":2,"componentType":5123,"count":3,"type":"SCALAR"}}
              ],
              "materials":[{{"pbrMetallicRoughness":{{"metallicFactor":0.25,"roughnessFactor":0.75}}}}],
              "meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"TEXCOORD_0":1}},"indices":2,"material":0}}]}}],
              "nodes":[{{"mesh":0}}],
              "scenes":[{{"nodes":[0]}}],
              "scene":0
            }}"#
        )
        .into_bytes();
        while json.len() % 4 != 0 {
            json.push(b' ');
        }

        let total_len = 12 + 8 + json.len() + 8 + bin.len();
        let mut glb = Vec::with_capacity(total_len);
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes());
        glb.extend_from_slice(&(total_len as u32).to_le_bytes());
        glb.extend_from_slice(&(json.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json);
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\0");
        glb.extend_from_slice(&bin);
        glb
    }

    #[test]
    fn imports_binary_gltf_geometry_and_material() {
        let asset = import_glb(&single_triangle_glb()).expect("valid GLB");
        assert_eq!(asset.mesh.vertices.len(), 3);
        assert_eq!(asset.mesh.indices, [0, 1, 2]);
        assert_eq!(asset.mesh.vertices[1].position, [1.0, 0.0, 0.0]);
        assert_eq!(asset.mesh.vertices[2].uv, [0.0, 1.0]);
        assert_eq!(asset.material.metallic, 0.25);
        assert_eq!(asset.material.roughness, 0.75);
        assert!(asset.base_color_texture.is_none());
        assert!(asset.metallic_roughness_texture.is_none());
        assert!(asset.normal_texture.is_none());
    }

    #[test]
    fn parses_disney_and_auxiliary_fields() {
        let materials = import_gltf_materials(
            br#"{
              "asset": {"version": "2.0"},
              "extensionsUsed": [
                "KHR_materials_anisotropy",
                "KHR_materials_clearcoat",
                "KHR_materials_emissive_strength",
                "KHR_materials_ior",
                "KHR_materials_sheen",
                "KHR_materials_specular",
                "KHR_materials_transmission"
              ],
              "images": [
                {"uri": "base.png"}, {"uri": "mr.png"}, {"uri": "normal.png"},
                {"uri": "occlusion.png"}, {"uri": "emissive.png"}
              ],
              "textures": [
                {"source": 0}, {"source": 1}, {"source": 2},
                {"source": 3}, {"source": 4}
              ],
              "materials": [{
                "name": "Coated glass",
                "alphaMode": "MASK",
                "alphaCutoff": 0.33,
                "doubleSided": true,
                "emissiveFactor": [0.1, 0.2, 0.3],
                "normalTexture": {"index": 2},
                "occlusionTexture": {"index": 3},
                "emissiveTexture": {"index": 4},
                "pbrMetallicRoughness": {
                  "baseColorFactor": [0.2, 0.3, 0.4, 0.6],
                  "metallicFactor": 0.7,
                  "roughnessFactor": 0.25,
                  "baseColorTexture": {"index": 0},
                  "metallicRoughnessTexture": {"index": 1}
                },
                "extensions": {
                  "KHR_materials_anisotropy": {"anisotropyStrength": 0.7},
                  "KHR_materials_clearcoat": {
                    "clearcoatFactor": 0.6,
                    "clearcoatRoughnessFactor": 0.2
                  },
                  "KHR_materials_emissive_strength": {"emissiveStrength": 2.5},
                  "KHR_materials_ior": {"ior": 1.8},
                  "KHR_materials_sheen": {"sheenColorFactor": [0.1, 0.4, 0.2]},
                  "KHR_materials_specular": {"specularFactor": 0.8},
                  "KHR_materials_transmission": {"transmissionFactor": 0.75}
                }
              }]
            }"#,
        )
        .expect("valid glTF material");

        let material = &materials[0];
        assert_eq!(material.name.as_deref(), Some("Coated glass"));
        assert_eq!(material.base_color, [0.2, 0.3, 0.4]);
        assert_eq!(material.metallic, 0.7);
        assert_eq!(material.roughness, 0.25);
        assert!((material.specular - 0.81632656).abs() < 1e-6);
        assert_eq!(material.anisotropic, 0.7);
        assert_eq!(material.sheen, 0.4);
        assert_eq!(material.clearcoat, 0.6);
        assert_eq!(material.clearcoat_gloss, 0.8);

        let auxiliary = material.auxiliary;
        assert_eq!(auxiliary.opacity, 0.6);
        assert_eq!(auxiliary.alpha_mode, AlphaMode::Mask);
        assert_eq!(auxiliary.alpha_cutoff, Some(0.33));
        assert!(auxiliary.double_sided);
        assert_eq!(auxiliary.emissive, [0.1, 0.2, 0.3]);
        assert_eq!(auxiliary.emissive_strength, 2.5);
        assert_eq!(auxiliary.ior, 1.8);
        assert_eq!(auxiliary.transmission, 0.75);
        assert_eq!(
            auxiliary.textures,
            MaterialTextures {
                base_color: true,
                metallic_roughness: true,
                normal: true,
                occlusion: true,
                emissive: true,
            }
        );
    }

    #[test]
    fn applies_gltf_defaults() {
        let materials = import_gltf_materials(br#"{"asset":{"version":"2.0"},"materials":[{}]}"#)
            .expect("valid default material");
        let material = &materials[0];
        assert_eq!(material.base_color, [1.0; 3]);
        assert_eq!(material.metallic, 1.0);
        assert_eq!(material.roughness, 1.0);
        assert!((material.specular - 0.5).abs() < 1e-6);
        assert_eq!(material.auxiliary, Auxiliary::default());
    }

    #[test]
    fn rejects_malformed_raw_extension_values() {
        let error = import_gltf_materials(
            br#"{
              "asset":{"version":"2.0"},
              "materials":[{
                "extensions":{"KHR_materials_clearcoat":{"clearcoatFactor":"high"}}
              }]
            }"#,
        )
        .expect_err("malformed factor must fail");
        assert!(matches!(
            error,
            GltfImportError::MalformedExtension {
                extension: "KHR_materials_clearcoat",
                field: "clearcoatFactor",
                ..
            }
        ));
    }
}
