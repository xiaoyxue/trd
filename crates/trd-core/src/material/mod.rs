//! Material models — plain data, no GPU types.
//!
//! `Material` / [`DisneyMaterial`] and friends contain **no `wgpu` and no
//! `bytemuck`**: they are the CPU-side description of a surface, siblings of
//! [`crate::mesh`], [`crate::texture`] and [`crate::camera`] rather than part of
//! the render backend. `render/pbr.rs` is what turns a material into GPU bytes.

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
