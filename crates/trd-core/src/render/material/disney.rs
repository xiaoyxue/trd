use std::collections::BTreeMap;

/// The glTF alpha-compositing mode preserved alongside a Disney material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    #[default]
    Opaque,
    Mask,
    Blend,
}

/// Presence flags for the five core glTF material texture slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MaterialTextures {
    pub base_color: bool,
    pub metallic_roughness: bool,
    pub normal: bool,
    pub occlusion: bool,
    pub emissive: bool,
}

/// glTF material data without a direct input in the current Disney shader.
///
/// The offline importer preserves these values now so later shading slices do
/// not need to change the material model or reparse source assets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Auxiliary {
    pub opacity: f32,
    pub alpha_mode: AlphaMode,
    pub alpha_cutoff: Option<f32>,
    pub double_sided: bool,
    pub emissive: [f32; 3],
    pub emissive_strength: f32,
    pub ior: f32,
    pub transmission: f32,
    pub textures: MaterialTextures,
}

impl Default for Auxiliary {
    fn default() -> Self {
        Self {
            opacity: 1.0,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: None,
            double_sided: false,
            emissive: [0.0; 3],
            emissive_strength: 1.0,
            ior: 1.5,
            transmission: 0.0,
            textures: MaterialTextures::default(),
        }
    }
}

/// Disney 2012 principled-BRDF surface parameters plus preserved import data.
///
/// Only the eleven surface parameters are consumed by `disney.wgsl`.
/// [`Auxiliary`] and [`sources`](Self::sources) remain CPU-side until their
/// corresponding visual slices land.
#[derive(Debug, Clone, PartialEq)]
pub struct DisneyMaterial {
    pub name: Option<String>,
    /// Linear-RGB tint multiplied onto the sampled albedo.
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub subsurface: f32,
    pub specular: f32,
    pub roughness: f32,
    pub specular_tint: f32,
    pub anisotropic: f32,
    pub sheen: f32,
    pub sheen_tint: f32,
    pub clearcoat: f32,
    pub clearcoat_gloss: f32,
    pub auxiliary: Auxiliary,
    /// Per-parameter provenance recorded by offline importers.
    pub sources: BTreeMap<String, String>,
}

impl Default for DisneyMaterial {
    fn default() -> Self {
        Self {
            name: None,
            base_color: [1.0; 3],
            metallic: 0.0,
            subsurface: 0.0,
            specular: 0.5,
            roughness: 0.5,
            specular_tint: 0.0,
            anisotropic: 0.0,
            sheen: 0.0,
            sheen_tint: 0.5,
            clearcoat: 0.0,
            clearcoat_gloss: 1.0,
            auxiliary: Auxiliary::default(),
            sources: BTreeMap::new(),
        }
    }
}

impl DisneyMaterial {
    /// A shiny metal preset used by the existing PBR demos.
    pub fn metal() -> Self {
        Self {
            metallic: 1.0,
            roughness: 0.28,
            ..Self::default()
        }
    }
}
