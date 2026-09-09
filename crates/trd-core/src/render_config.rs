//! Document-wide appearance settings, independent of the delivery surface.

use serde::{Deserialize, Deserializer, Serialize};

/// Shared by every row and record batch in a scene document.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RenderConfig {
    #[serde(deserialize_with = "RenderConfig::deserialize_object")]
    pub shadow: ShadowConfig,
}

impl RenderConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let mut deserializer = serde_json::Deserializer::from_str(json);
        let config = Self::deserialize_object(&mut deserializer)?;
        deserializer.end()?;
        Ok(config)
    }

    fn deserialize_object<'de, D, T>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
        T: Deserialize<'de>,
    {
        struct ObjectVisitor<T>(std::marker::PhantomData<T>);
        impl<'de, T: Deserialize<'de>> serde::de::Visitor<'de> for ObjectVisitor<T> {
            type Value = T;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a configuration object")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
                T::deserialize(serde::de::value::MapAccessDeserializer::new(map))
            }
        }
        deserializer.deserialize_map(ObjectVisitor(std::marker::PhantomData))
    }

    pub fn to_json(self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ShadowConfig {
    pub enable: bool,
    pub shadow_type: ShadowType,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enable: true,
            shadow_type: ShadowType::Blob,
        }
    }
}

/// Only implemented techniques can enter the renderer's typed configuration.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowType {
    #[default]
    Blob,
}

impl<'de> Deserialize<'de> for ShadowType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "blob" => Ok(Self::Blob),
            "shadow_map" => Err(serde::de::Error::custom(
                "unsupported shadow type \"shadow_map\"; only \"blob\" is implemented",
            )),
            _ => Err(serde::de::Error::unknown_variant(&value, &["blob"])),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_fields_default_to_enabled_blob() {
        for json in ["{}", r#"{"shadow":{}}"#, r#"{"shadow":{"enable":true}}"#] {
            assert_eq!(
                serde_json::from_str::<RenderConfig>(json).unwrap(),
                RenderConfig::default()
            );
        }
        let config: RenderConfig = serde_json::from_str(r#"{"shadow":{"enable":false}}"#).unwrap();
        assert!(!config.shadow.enable);
        assert_eq!(config.shadow.shadow_type, ShadowType::Blob);
        assert_eq!(
            serde_json::from_str::<RenderConfig>(&serde_json::to_string(&config).unwrap()).unwrap(),
            config
        );
    }

    #[test]
    fn malformed_or_unsupported_configuration_never_defaults() {
        for json in [
            "null",
            "[]",
            "{",
            r#"{"shadow":[]}"#,
            r#"{"shaow":{}}"#,
            r#"{"shadow":null}"#,
            r#"{"shadow":{"enable":null}}"#,
            r#"{"shadow":{"enable":"false"}}"#,
            r#"{"shadow":{"enabled":false}}"#,
            r#"{"shadow":{"shadow_type":null}}"#,
            r#"{"shadow":{"shadow_type":"blobl"}}"#,
            r#"{"shadow":{"shadow_type":"shadow_map"}}"#,
            r#"{"shadow":{"enable":false,"shadow_type":"shadow_map"}}"#,
        ] {
            assert!(RenderConfig::from_json(json).is_err(), "{json}");
        }
        assert!(
            serde_json::from_str::<RenderConfig>(r#"{"shadow":{"shadow_type":"shadow_map"}}"#)
                .unwrap_err()
                .to_string()
                .contains("unsupported shadow type")
        );
    }
}
