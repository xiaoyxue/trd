use std::ops::Range;

use super::{GltfAsset, Mesh};
use crate::{DisneyMaterial, ImageTexture};

/// One material's share of a mesh: a contiguous run of its triangle indices,
/// drawn with this material and these maps (#394).
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialPart {
    /// Into [`Mesh::indices`]; a multiple of three.
    pub indices: Range<u32>,
    pub material: DisneyMaterial,
    pub base_color_texture: Option<ImageTexture>,
    pub metallic_roughness_texture: Option<ImageTexture>,
    pub normal_texture: Option<ImageTexture>,
}

impl MaterialPart {
    /// The single untextured part covering every triangle of `mesh`.
    pub fn whole(mesh: &Mesh, material: DisneyMaterial) -> Self {
        Self {
            indices: 0..mesh.indices.len() as u32,
            material,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshAssetSource {
    Embedded,
    Gltf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshReference {
    pub path: Option<String>,
    pub url: Option<String>,
}

impl MeshReference {
    pub fn new(path: Option<String>, url: Option<String>) -> Option<Self> {
        let path = path.filter(|value| !value.is_empty());
        let url = url.filter(|value| !value.is_empty());
        (path.is_some() || url.is_some()).then_some(Self { path, url })
    }

    pub fn display(&self) -> &str {
        self.path
            .as_deref()
            .or(self.url.as_deref())
            .unwrap_or("<missing glTF reference>")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeshAsset {
    pub mesh_id: Option<u32>,
    pub source: MeshAssetSource,
    pub mesh: Mesh,
    /// Never empty; see [`GltfAsset::parts`].
    pub parts: Vec<MaterialPart>,
}

impl MeshAsset {
    pub fn embedded(mesh: Mesh, material: DisneyMaterial) -> Self {
        Self {
            mesh_id: None,
            source: MeshAssetSource::Embedded,
            parts: vec![MaterialPart::whole(&mesh, material)],
            mesh,
        }
    }

    pub fn embedded_with_id(mesh_id: u32, mesh: Mesh, material: DisneyMaterial) -> Self {
        Self {
            mesh_id: Some(mesh_id),
            ..Self::embedded(mesh, material)
        }
    }

    pub fn from_gltf_with_id(mesh_id: u32, asset: GltfAsset) -> Self {
        Self {
            mesh_id: Some(mesh_id),
            ..Self::from(asset)
        }
    }

    pub fn mesh_id_or(&self, fallback: u32) -> u32 {
        self.mesh_id.unwrap_or(fallback)
    }

    /// The first part — the object's material when it has only one.
    pub fn primary(&self) -> &MaterialPart {
        &self.parts[0]
    }

    pub fn primary_mut(&mut self) -> &mut MaterialPart {
        &mut self.parts[0]
    }
}

impl From<GltfAsset> for MeshAsset {
    fn from(asset: GltfAsset) -> Self {
        Self {
            mesh_id: None,
            source: MeshAssetSource::Gltf,
            mesh: asset.mesh,
            parts: asset.parts,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeshResource {
    Resolved(Box<MeshAsset>),
    Gltf(MeshReference),
}
