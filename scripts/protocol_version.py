"""The sole scene-wire version emitted by Python producers."""

import json

PROTOCOL_VERSION = "0.0.7"
RENDER_CONFIG_KEY = b"trd.render.config"


def render_config_metadata(enable: bool = True) -> dict[bytes, bytes]:
    """One params-schema value, shared by all batches and sparse rows."""
    config = {"shadow": {"enable": enable, "shadow_type": "blob"}}
    return {RENDER_CONFIG_KEY: json.dumps(config, separators=(",", ":")).encode()}


def validate_render_config_metadata(metadata: dict[bytes, bytes]) -> None:
    config = json.loads(metadata.get(RENDER_CONFIG_KEY, b"{}"))
    if not isinstance(config, dict) or config.keys() - {"shadow"}:
        raise ValueError("trd.render.config must be an object with only the shadow field")
    shadow = config.get("shadow", {})
    if not isinstance(shadow, dict) or shadow.keys() - {"enable", "shadow_type"}:
        raise ValueError("shadow must be an object with only enable and shadow_type fields")
    if type(shadow.get("enable", True)) is not bool:
        raise ValueError("shadow.enable must be a boolean")
    if shadow.get("shadow_type", "blob") != "blob":
        raise ValueError("unsupported shadow type; only blob is implemented")
