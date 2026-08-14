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

/// The HDR environment probe **as a light** (#182).
///
/// The probe's yaw used to exist twice — as per-mesh
/// [`ImageBasedLighting::rotation`](crate::ImageBasedLighting) driving
/// reflections and as `EnvironmentBackground::rotation` driving the visible sky
/// — with nothing keeping them equal, so an object could reflect an environment
/// oriented differently from the sky behind it. (`trd-gui` had to hand-sync
/// them.) It lives here **once** instead, which also makes "IBL is a kind of
/// light" hold structurally.
///
/// Added by *containment* rather than by turning [`Light`] into an enum: the
/// shader packs directional/point lights as fixed `vec4` arrays and the
/// environment is a singleton probe, so an enum would fight the uniform layout
/// for no gain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvironmentLight {
    /// Scene-wide gain on the probe's contribution. Composes **multiplicatively**
    /// with each object's [`ImageBasedLighting::intensity`](crate::ImageBasedLighting):
    /// the effective gain is `mesh.intensity * scene.intensity`, so the default
    /// `1.0` leaves every existing per-object value untouched.
    pub intensity: f32,
    /// Yaw applied to the probe, in radians — the **single** source of truth for
    /// both the reflections and the sky drawn behind them.
    pub rotation: f32,
}

impl Default for EnvironmentLight {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation: 0.0,
        }
    }
}

/// Scene lighting controls shared by every Disney-shaded object.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lighting {
    /// Constant fill multiplied by the surface base color.
    pub ambient: f32,
    /// Gain applied to every light in the fixed rig.
    pub scale: f32,
    /// The HDR probe as a light: its scene-wide gain and its yaw.
    pub environment: EnvironmentLight,
}

impl Lighting {
    /// The default rig: `ambient` fill, rig `scale`, and an unrotated probe at
    /// unit gain.
    pub const DEFAULT: Self = Self {
        ambient: 0.12,
        scale: 2.5,
        environment: EnvironmentLight {
            intensity: 1.0,
            rotation: 0.0,
        },
    };
}

impl Default for Lighting {
    fn default() -> Self {
        Self::DEFAULT
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
