use super::{GltfAsset, Mesh};
use crate::{DisneyMaterial, ImageTexture};

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
    pub mesh: Mesh,
    pub material: DisneyMaterial,
    pub base_color_texture: Option<ImageTexture>,
    pub metallic_roughness_texture: Option<ImageTexture>,
    pub normal_texture: Option<ImageTexture>,
}

impl MeshAsset {
    pub fn embedded(mesh: Mesh, material: DisneyMaterial) -> Self {
        Self {
            mesh,
            material,
            base_color_texture: None,
            metallic_roughness_texture: None,
            normal_texture: None,
        }
    }
}

impl From<GltfAsset> for MeshAsset {
    fn from(asset: GltfAsset) -> Self {
        Self {
            mesh: asset.mesh,
            material: asset.material,
            base_color_texture: asset.base_color_texture,
            metallic_roughness_texture: asset.metallic_roughness_texture,
            normal_texture: asset.normal_texture,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MeshResource {
    Resolved(Box<MeshAsset>),
    Gltf(MeshReference),
}
