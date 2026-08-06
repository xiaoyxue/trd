mod disney;

pub use disney::{AlphaMode, Auxiliary, DisneyMaterial, MaterialTextures};

/// Closed material dispatch extended by later shading-model slices.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Material {
    Disney(DisneyMaterial),
}

impl Default for Material {
    fn default() -> Self {
        Self::Disney(DisneyMaterial::default())
    }
}
