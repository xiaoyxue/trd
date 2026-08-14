/// A world-space directional light.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// Direction in which the light travels.
    pub direction: [f32; 3],
    pub intensity: f32,
}

impl Light {
    pub(crate) const fn to_uniform(self) -> [f32; 4] {
        [
            self.direction[0],
            self.direction[1],
            self.direction[2],
            self.intensity,
        ]
    }
}

/// A world-space point light with inverse-square falloff in the shader.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointLight {
    pub position: [f32; 3],
    pub intensity: f32,
}

impl PointLight {
    pub(crate) const fn to_uniform(self) -> [f32; 4] {
        [
            self.position[0],
            self.position[1],
            self.position[2],
            self.intensity,
        ]
    }
}

/// Scene lighting controls shared by every Disney-shaded object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    /// Constant fill multiplied by the surface base color.
    pub ambient: f32,
    /// Gain applied to every light in the fixed rig.
    pub scale: f32,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            ambient: 0.12,
            scale: 2.5,
        }
    }
}

/// The existing key/fill/rim rig expressed as typed data.
pub(crate) const DEFAULT_LIGHTS: [Light; 3] = [
    Light {
        direction: [-0.5, -0.85, -0.55],
        intensity: 1.0,
    },
    Light {
        direction: [0.8, -0.25, 0.35],
        intensity: 0.4,
    },
    Light {
        direction: [0.25, -0.3, 0.9],
        intensity: 0.55,
    },
];

/// Point-light slots remain empty until scene-authored lights are introduced.
pub(crate) const DEFAULT_POINT_LIGHTS: [PointLight; 0] = [];
