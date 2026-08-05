//! glTF material import into trd's typed Disney model.

use std::collections::BTreeMap;

use crate::{AlphaMode, Auxiliary, DisneyMaterial, MaterialTextures};
use thiserror::Error;

const DEFAULT_IOR: f32 = 1.5;

#[derive(Debug, Error)]
pub enum GltfImportError {
    #[error("invalid glTF: {0}")]
    Gltf(#[from] gltf_rs::Error),
    #[error("glTF extension `{extension}` field `{field}` must be {expected}")]
    MalformedExtension {
        extension: &'static str,
        field: &'static str,
        expected: &'static str,
    },
}

/// Parses every explicit glTF material without loading geometry or image bytes.
pub fn import_gltf_materials(bytes: &[u8]) -> Result<Vec<DisneyMaterial>, GltfImportError> {
    let gltf = gltf_rs::Gltf::from_slice(bytes)?;
    gltf.document.materials().map(import_material).collect()
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
