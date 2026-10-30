//! glTF material import into trd's typed Disney model.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    AlphaMode, Auxiliary, DisneyMaterial, ImageTexture, MaterialTextures, Matrix4, Mesh,
    MeshShading, Normal3, Point3, TextureError, Transform, Vector3, Vertex,
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
    #[error("glTF scene draws no mesh primitive")]
    MissingPrimitive,
    #[error("glTF import currently requires every primitive to share one material; found {count}")]
    MultipleMaterials { count: usize },
    #[error("glTF node {0} is reachable twice; the node graph must be a tree")]
    NodeGraph(usize),
    #[error("glTF scene has more vertices than a u32 index can address")]
    TooManyVertices,
    #[error("glTF primitive index {index} is out of range for its {vertices} vertices")]
    IndexOutOfRange { index: u32, vertices: usize },
    #[error("glTF primitive mode {0:?} is unsupported (expected triangles)")]
    PrimitiveMode(gltf_rs::mesh::Mode),
    #[error("glTF primitive has no POSITION attribute")]
    MissingPositions,
    #[error("glTF attribute {attribute} has {actual} values for {expected} positions")]
    AttributeLength {
        attribute: &'static str,
        expected: usize,
        actual: usize,
    },
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

/// Parses a binary glTF scene into one mesh, its Disney material, and its
/// embedded textures.
///
/// The default scene (else the first; else every mesh, for scene-less files)
/// is walked and each node's world transform baked into its primitives, which
/// are merged into one [`Mesh`] in traversal order. Skins and morph targets are
/// ignored, so the bind pose is drawn. The asset stays one object with one
/// material: primitives using different materials are rejected (#394).
pub fn import_glb(bytes: &[u8]) -> Result<GltfAsset, GltfImportError> {
    let gltf = gltf_rs::Gltf::from_slice(bytes)?;
    let blob = gltf.blob.as_deref().ok_or(GltfImportError::MissingBlob)?;
    let placed = scene_primitives(&gltf.document)?;
    let (first, _) = placed.first().ok_or(GltfImportError::MissingPrimitive)?;
    let materials: BTreeSet<Option<usize>> = placed
        .iter()
        .map(|(primitive, _)| primitive.material().index())
        .collect();
    if materials.len() > 1 {
        return Err(GltfImportError::MultipleMaterials {
            count: materials.len(),
        });
    }

    let mut merged = MergedMesh::default();
    for (primitive, world) in &placed {
        merged.append(primitive, *world, blob)?;
    }
    if merged.vertices.is_empty() {
        return Err(GltfImportError::MissingPrimitive);
    }

    let source_material = first.material();
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
        mesh: merged.finish(),
        material,
        base_color_texture,
        metallic_roughness_texture,
        normal_texture,
    })
}

/// Every primitive the scene draws, in depth-first node order, with the world
/// matrix of the node that instantiates it.
fn scene_primitives(
    document: &gltf_rs::Document,
) -> Result<Vec<(gltf_rs::Primitive<'_>, Matrix4)>, GltfImportError> {
    let Some(scene) = document
        .default_scene()
        .or_else(|| document.scenes().next())
    else {
        return Ok(document
            .meshes()
            .flat_map(|mesh| mesh.primitives())
            .map(|primitive| (primitive, Matrix4::IDENTITY))
            .collect());
    };
    // A node reached twice means a cycle or a second parent, both invalid; the
    // check also bounds the walk on a malicious file.
    let mut visited = vec![false; document.nodes().len()];
    let mut stack: Vec<_> = scene
        .nodes()
        .map(|node| (node, Matrix4::IDENTITY))
        .collect();
    stack.reverse();
    let mut placed = Vec::new();
    while let Some((node, parent)) = stack.pop() {
        if std::mem::replace(&mut visited[node.index()], true) {
            return Err(GltfImportError::NodeGraph(node.index()));
        }
        let local = Matrix4::from_glam(glam::Mat4::from_cols_array_2d(&node.transform().matrix()));
        let world = parent * local;
        if let Some(mesh) = node.mesh() {
            placed.extend(mesh.primitives().map(|primitive| (primitive, world)));
        }
        // Pushed reversed so the first child pops first.
        let siblings = stack.len();
        stack.extend(node.children().map(|child| (child, world)));
        stack[siblings..].reverse();
    }
    Ok(placed)
}

/// Primitives concatenated into one indexed mesh, baked into world space.
#[derive(Default)]
struct MergedMesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    /// `None` once any primitive lacks `NORMAL`: the merged mesh then falls
    /// back to derived smooth normals as a whole (#393 makes these flat).
    normals: Option<Vec<[f32; 3]>>,
    /// `None` once any primitive lacks `TANGENT`, so all tangents are derived.
    tangents: Option<Vec<[f32; 4]>>,
}

impl MergedMesh {
    fn append(
        &mut self,
        primitive: &gltf_rs::Primitive<'_>,
        world: Matrix4,
        blob: &[u8],
    ) -> Result<(), GltfImportError> {
        if primitive.mode() != gltf_rs::mesh::Mode::Triangles {
            return Err(GltfImportError::PrimitiveMode(primitive.mode()));
        }
        let reader = primitive.reader(|buffer| match buffer.source() {
            gltf_rs::buffer::Source::Bin => Some(blob),
            gltf_rs::buffer::Source::Uri(_) => None,
        });
        let mut positions: Vec<[f32; 3]> = reader
            .read_positions()
            .ok_or(GltfImportError::MissingPositions)?
            .collect();
        let tex_coords: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|coords| coords.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0; 2]; positions.len()]);
        let colors: Vec<[f32; 3]> = reader
            .read_colors(0)
            .map(|colors| colors.into_rgb_f32().collect())
            .unwrap_or_else(|| vec![[1.0; 3]; positions.len()]);
        let mut normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(Iterator::collect);
        let mut tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(Iterator::collect);
        for (attribute, count) in [
            ("TEXCOORD_0", Some(tex_coords.len())),
            ("COLOR_0", Some(colors.len())),
            ("NORMAL", normals.as_ref().map(Vec::len)),
            ("TANGENT", tangents.as_ref().map(Vec::len)),
        ] {
            if let Some(actual) = count.filter(|count| *count != positions.len()) {
                return Err(GltfImportError::AttributeLength {
                    attribute,
                    expected: positions.len(),
                    actual,
                });
            }
        }
        let mut indices: Vec<u32> = reader
            .read_indices()
            .map(|indices| indices.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        // Once merged, a bad index would silently land in another primitive.
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= positions.len())
        {
            return Err(GltfImportError::IndexOutOfRange {
                index,
                vertices: positions.len(),
            });
        }

        let determinant = world.determinant();
        if determinant == 0.0 || !determinant.is_finite() {
            // A zero-scale node is how glTF hides geometry; it has no area.
            return Ok(());
        }
        // Identity skips the arithmetic so single-node assets keep their exact
        // authored floats (and the goldens their pixels).
        if world != Matrix4::IDENTITY {
            let transform = Transform::from_matrix(world);
            for position in &mut positions {
                *position = transform
                    .transform_point(Point3::from_array(*position))
                    .to_array();
            }
            for normal in normals.iter_mut().flatten() {
                *normal = transform
                    .transform_normal(Normal3::from_array(*normal))
                    .normalize_or_zero()
                    .to_array();
            }
            let mirrored = determinant < 0.0;
            for tangent in tangents.iter_mut().flatten() {
                let [x, y, z] = transform
                    .transform_vector(Vector3::new(tangent[0], tangent[1], tangent[2]))
                    .normalize_or_zero()
                    .to_array();
                let w = if mirrored { -tangent[3] } else { tangent[3] };
                *tangent = [x, y, z, w];
            }
            // A mirror flips the triangles' facing; restore counter-clockwise.
            if mirrored {
                for triangle in indices.as_chunks_mut::<3>().0 {
                    triangle.swap(1, 2);
                }
            }
        }

        let first_primitive = self.vertices.is_empty();
        let base =
            u32::try_from(self.vertices.len()).map_err(|_| GltfImportError::TooManyVertices)?;
        self.vertices
            .extend(positions.into_iter().zip(tex_coords).zip(colors).map(
                |((position, uv), color)| Vertex {
                    position,
                    color,
                    uv,
                },
            ));
        self.indices
            .extend(indices.into_iter().map(|index| base + index));
        self.normals = merge_attribute(first_primitive, self.normals.take(), normals);
        self.tangents = merge_attribute(first_primitive, self.tangents.take(), tangents);
        Ok(())
    }

    fn finish(self) -> Mesh {
        Mesh {
            shading: self.normals.map(|normals| MeshShading {
                normals,
                tangents: self.tangents.unwrap_or_default(),
            }),
            vertices: self.vertices,
            indices: self.indices,
        }
    }
}

/// An attribute survives the merge only if every primitive so far carries it.
fn merge_attribute<T>(first: bool, merged: Option<Vec<T>>, next: Option<Vec<T>>) -> Option<Vec<T>> {
    if first {
        return next;
    }
    let mut merged = merged?;
    merged.extend(next?);
    Some(merged)
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

/// Builds small GLBs in test code: accessors are appended to one BIN buffer and
/// the caller supplies the rest of the glTF JSON.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestGlb {
    bin: Vec<u8>,
    views: Vec<serde_json::Value>,
    accessors: Vec<serde_json::Value>,
}

#[cfg(test)]
impl TestGlb {
    /// A float VECn accessor (with min/max, which `POSITION` requires).
    pub(crate) fn floats<const N: usize>(&mut self, values: &[[f32; N]]) -> usize {
        let mut min = [f32::INFINITY; N];
        let mut max = [f32::NEG_INFINITY; N];
        for value in values {
            for (axis, component) in value.iter().enumerate() {
                min[axis] = min[axis].min(*component);
                max[axis] = max[axis].max(*component);
            }
        }
        let bytes: Vec<u8> = values
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let kind = ["SCALAR", "VEC2", "VEC3", "VEC4"][N - 1];
        self.accessor(
            &bytes,
            serde_json::json!({
                "componentType": 5126, "count": values.len(), "type": kind,
                "min": min.to_vec(), "max": max.to_vec()
            }),
        )
    }

    pub(crate) fn indices(&mut self, values: &[u16]) -> usize {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.accessor(
            &bytes,
            serde_json::json!({"componentType": 5123, "count": values.len(), "type": "SCALAR"}),
        )
    }

    fn accessor(&mut self, bytes: &[u8], mut accessor: serde_json::Value) -> usize {
        accessor["bufferView"] = self.views.len().into();
        self.views.push(serde_json::json!({
            "buffer": 0, "byteOffset": self.bin.len(), "byteLength": bytes.len()
        }));
        self.bin.extend_from_slice(bytes);
        while !self.bin.len().is_multiple_of(4) {
            self.bin.push(0);
        }
        self.accessors.push(accessor);
        self.accessors.len() - 1
    }

    /// Packs `document` (meshes, nodes, scenes, materials…) with the accessors.
    pub(crate) fn finish(self, mut document: serde_json::Value) -> Vec<u8> {
        document["asset"] = serde_json::json!({"version": "2.0"});
        document["buffers"] = serde_json::json!([{"byteLength": self.bin.len()}]);
        document["bufferViews"] = self.views.into();
        document["accessors"] = self.accessors.into();
        let mut json = serde_json::to_vec(&document).unwrap();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let total_len = 12 + 8 + json.len() + 8 + self.bin.len();
        let mut glb = b"glTF".to_vec();
        glb.extend(2u32.to_le_bytes());
        glb.extend(u32::try_from(total_len).unwrap().to_le_bytes());
        glb.extend(u32::try_from(json.len()).unwrap().to_le_bytes());
        glb.extend(b"JSON");
        glb.extend(json);
        glb.extend(u32::try_from(self.bin.len()).unwrap().to_le_bytes());
        glb.extend(b"BIN\0");
        glb.extend(self.bin);
        glb
    }
}

/// Two one-triangle meshes on two root nodes, the second offset by `+x 10`.
#[cfg(test)]
pub(crate) fn two_mesh_glb() -> Vec<u8> {
    let mut glb = TestGlb::default();
    let positions = glb.floats(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let indices = glb.indices(&[0, 1, 2]);
    let primitive = serde_json::json!({"attributes": {"POSITION": positions}, "indices": indices});
    glb.finish(serde_json::json!({
        "meshes": [{"primitives": [primitive]}, {"primitives": [primitive]}],
        "nodes": [{"mesh": 0}, {"mesh": 1, "translation": [10.0, 0.0, 0.0]}],
        "scenes": [{"nodes": [0, 1]}],
        "scene": 0
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TRIANGLE: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    /// One triangle primitive with `+z` normals and optional tangents.
    fn triangle_primitive(glb: &mut TestGlb, normals: bool, tangents: bool) -> serde_json::Value {
        let mut attributes = json!({"POSITION": glb.floats(&TRIANGLE)});
        if normals {
            attributes["NORMAL"] = glb.floats(&[[0.0, 0.0, 1.0]; 3]).into();
        }
        if tangents {
            attributes["TANGENT"] = glb.floats(&[[1.0, 0.0, 0.0, 1.0]; 3]).into();
        }
        json!({"attributes": attributes, "indices": glb.indices(&[0, 1, 2])})
    }

    /// One triangle mesh drawn by the given nodes (node 0 is the scene root).
    fn triangle_scene(nodes: serde_json::Value) -> GltfAsset {
        let mut glb = TestGlb::default();
        let primitive = triangle_primitive(&mut glb, true, true);
        import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [primitive]}],
            "nodes": nodes,
            "scenes": [{"nodes": [0]}],
            "scene": 0
        })))
        .expect("valid GLB")
    }

    fn positions(asset: &GltfAsset) -> Vec<[f32; 3]> {
        asset.mesh.vertices.iter().map(|v| v.position).collect()
    }

    fn assert_close(actual: [f32; 3], expected: [f32; 3]) {
        for (a, e) in actual.into_iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "{actual:?} != {expected:?}");
        }
    }

    #[test]
    fn imports_every_mesh_of_the_default_scene() {
        let asset = import_glb(&two_mesh_glb()).expect("valid GLB");
        assert_eq!(asset.mesh.vertices.len(), 6);
        assert_eq!(asset.mesh.indices, [0, 1, 2, 3, 4, 5]);
        assert_eq!(asset.mesh.vertices[4].position, [11.0, 0.0, 0.0]);
    }

    #[test]
    fn bakes_node_hierarchy_transforms() {
        // Parent translates +y 2; the TRS child scales x2 then turns 90° about z.
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let trs = triangle_scene(json!([
            {"translation": [0.0, 2.0, 0.0], "children": [1]},
            {"mesh": 0, "scale": [2.0, 2.0, 2.0], "rotation": [0.0, 0.0, half, half]}
        ]));
        assert_close(trs.mesh.vertices[1].position, [0.0, 4.0, 0.0]);
        assert_close(trs.mesh.vertices[2].position, [-2.0, 2.0, 0.0]);

        // The same child authored as a column-major `matrix`.
        let matrix = triangle_scene(json!([
            {"translation": [0.0, 2.0, 0.0], "children": [1]},
            {"mesh": 0, "matrix": [
                0.0, 2.0, 0.0, 0.0,  -2.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 0.0,   0.0, 0.0, 0.0, 1.0
            ]}
        ]));
        for (a, b) in positions(&trs).into_iter().zip(positions(&matrix)) {
            assert_close(a, b);
        }
    }

    #[test]
    fn instanced_mesh_is_emitted_per_node() {
        let asset = triangle_scene(json!([
            {"children": [1, 2]},
            {"mesh": 0},
            {"mesh": 0, "translation": [0.0, 0.0, 5.0]}
        ]));
        assert_eq!(asset.mesh.vertices.len(), 6);
        assert_eq!(asset.mesh.vertices[3].position, [0.0, 0.0, 5.0]);
        assert_eq!(asset.mesh.indices[3..], [3, 4, 5]);
    }

    #[test]
    fn negative_determinant_reverses_winding() {
        let asset = triangle_scene(json!([{"mesh": 0, "scale": [-1.0, 1.0, 1.0]}]));
        assert_eq!(asset.mesh.indices, [0, 2, 1]);
        assert_eq!(asset.mesh.vertices[1].position, [-1.0, 0.0, 0.0]);
        let shading = asset.mesh.shading.expect("authored normals");
        assert_eq!(shading.tangents[0], [-1.0, 0.0, 0.0, -1.0]);
        // The mirror keeps +z normals, and the reversed winding faces +z too.
        assert_close(shading.normals[0], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn normals_use_inverse_transpose() {
        // Normal (1,1,0)/√2 on a plane x = -y; scaling x by 4 tilts the plane,
        // so the normal must turn toward +y, not toward +x.
        let mut glb = TestGlb::default();
        let n = std::f32::consts::FRAC_1_SQRT_2;
        let primitive = json!({"attributes": {
            "POSITION": glb.floats(&[[1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 1.0, 0.0]]),
            "NORMAL": glb.floats(&[[n, n, 0.0]; 3])
        }});
        let asset = import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [primitive]}],
            "nodes": [{"mesh": 0, "scale": [4.0, 1.0, 1.0]}],
            "scenes": [{"nodes": [0]}]
        })))
        .unwrap();
        let normal = asset.mesh.shading.unwrap().normals[0];
        let [a, b, c] = [0, 1, 2].map(|i| asset.mesh.vertices[i].position);
        for edge in [
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
            [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
        ] {
            let dot: f32 = edge.iter().zip(normal).map(|(e, n)| e * n).sum();
            assert!(
                dot.abs() < 1e-5,
                "normal {normal:?} not perpendicular to {edge:?}"
            );
        }
        assert!((normal.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(normal[1] > normal[0]);
    }

    #[test]
    fn partial_normals_drop_authored_shading() {
        let mut glb = TestGlb::default();
        let with = triangle_primitive(&mut glb, true, false);
        let without = triangle_primitive(&mut glb, false, false);
        let asset = import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [with, without]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}]
        })))
        .unwrap();
        assert_eq!(asset.mesh.vertices.len(), 6);
        assert!(asset.mesh.shading.is_none());
    }

    #[test]
    fn partial_tangents_are_derived_for_the_whole_mesh() {
        let mut glb = TestGlb::default();
        let with = triangle_primitive(&mut glb, true, true);
        let without = triangle_primitive(&mut glb, true, false);
        let asset = import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [with, without]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}]
        })))
        .unwrap();
        let shading = asset.mesh.shading.unwrap();
        assert_eq!(shading.normals.len(), 6);
        assert!(shading.tangents.is_empty());
    }

    #[test]
    fn rejects_primitives_with_different_materials() {
        let mut glb = TestGlb::default();
        let mut first = triangle_primitive(&mut glb, true, false);
        let mut second = triangle_primitive(&mut glb, true, false);
        first["material"] = 0.into();
        second["material"] = 1.into();
        let error = import_glb(&glb.finish(json!({
            "materials": [{}, {}],
            "meshes": [{"primitives": [first, second]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}]
        })))
        .expect_err("two materials");
        assert!(matches!(
            error,
            GltfImportError::MultipleMaterials { count: 2 }
        ));
    }

    #[test]
    fn scene_less_document_falls_back_to_all_meshes() {
        let mut glb = TestGlb::default();
        let primitive = triangle_primitive(&mut glb, false, false);
        let asset = import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [primitive]}, {"primitives": [primitive]}],
            "nodes": [{"mesh": 0, "translation": [9.0, 9.0, 9.0]}]
        })))
        .unwrap();
        assert_eq!(positions(&asset), [TRIANGLE, TRIANGLE].concat());
    }

    #[test]
    fn millimetre_scale_nodes_are_kept_and_zero_scale_nodes_dropped() {
        let asset = triangle_scene(json!([
            {"children": [1, 2]},
            {"mesh": 0, "scale": [0.001, 0.001, 0.001]},
            {"mesh": 0, "scale": [0.0, 0.0, 0.0]}
        ]));
        assert_eq!(asset.mesh.vertices.len(), 3);
        assert_close(asset.mesh.vertices[1].position, [0.001, 0.0, 0.0]);
        assert_close(asset.mesh.shading.unwrap().normals[0], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn rejects_an_index_past_its_own_primitive() {
        let mut glb = TestGlb::default();
        let positions = glb.floats(&TRIANGLE);
        let indices = glb.indices(&[0, 1, 3]);
        let primitive = json!({"attributes": {"POSITION": positions}, "indices": indices});
        let error = import_glb(&glb.finish(json!({
            // A second primitive would make index 3 resolve after merging.
            "meshes": [{"primitives": [primitive, primitive]}],
            "nodes": [{"mesh": 0}],
            "scenes": [{"nodes": [0]}]
        })))
        .expect_err("index 3 of 3 vertices");
        assert!(matches!(
            error,
            GltfImportError::IndexOutOfRange {
                index: 3,
                vertices: 3
            }
        ));
    }

    #[test]
    fn rejects_a_node_reachable_twice() {
        let mut glb = TestGlb::default();
        let primitive = triangle_primitive(&mut glb, false, false);
        let error = import_glb(&glb.finish(json!({
            "meshes": [{"primitives": [primitive]}],
            "nodes": [{"children": [1]}, {"mesh": 0}],
            "scenes": [{"nodes": [0, 1]}]
        })))
        .expect_err("node 1 has two parents");
        assert!(matches!(error, GltfImportError::NodeGraph(1)));
    }

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
