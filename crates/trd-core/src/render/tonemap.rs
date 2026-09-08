/// Tone-mapping operator applied to one object's linear PBR radiance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tonemap {
    #[default]
    Reinhard,
    Aces,
}

impl Tonemap {
    pub(crate) const fn from_wire(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Reinhard),
            1 => Some(Self::Aces),
            _ => None,
        }
    }

    pub(crate) const fn to_wire(self) -> u8 {
        match self {
            Self::Reinhard => 0,
            Self::Aces => 1,
        }
    }

    pub(crate) const fn to_uniform(self) -> f32 {
        match self {
            Self::Reinhard => 0.0,
            Self::Aces => 1.0,
        }
    }
}

/// Per-object PBR output transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToneMapping {
    pub operator: Tonemap,
    pub exposure: f32,
}

impl Default for ToneMapping {
    fn default() -> Self {
        Self {
            operator: Tonemap::Reinhard,
            exposure: 1.2,
        }
    }
}
