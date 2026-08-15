# Physically-based rendering (PBR)

`--pbr` shades meshes with the **Disney principled BRDF** (Burley 2012) instead of
flat per-vertex color or a plain textured lookup. The BRDF lives in
[`crates/trd-core/src/shader/pbr.wgsl`](../crates/trd-core/src/shader/pbr.wgsl) (reference:
[`ref/DisneyPBR/shader.frag`](../ref/DisneyPBR/shader.frag)); it lights the bound
albedo with a fixed **virtual light rig** (three directional lights — key, fill,
rim, in world space, so a spinning object is lit from changing angles) plus an
optional **HDR environment probe**, derives smooth per-vertex shading normals, and
tone-maps the linear radiance to the sRGB target.

`--pbr` requires a **texture table** for the albedo (like `--textured`), and
conflicts with `--wireframe` / `--textured`.

```sh
# Shiny metal bunny under an HDR environment probe, ACES tone-map:
examples/render.sh --cli --pbr --metallic 1 --roughness 0.3 --tonemap aces \
  --env assets/envmap/uffizi-large.hdr \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg \
  examples/frames.bunny_dolly.cg.jsonl output/bunny_pbr.gif 512 512 20
```

## CLI parameters

`trd-cli` (and `render.sh` / `render.ps1`, which pass them through) expose the most
useful material + lighting controls as flags. All are ignored unless `--pbr` is
set.

| Flag | Default | Range | Meaning |
|---|---|---|---|
| `--pbr` | off | — | Enable the Disney principled BRDF path. |
| `--metallic` | `0.0` | `0..1` | 0 = dielectric, 1 = metal (kills the diffuse lobe, tints reflection). |
| `--roughness` | `0.35` | `0..1` | 0 = mirror, 1 = fully rough. |
| `--specular` | `0.5` | `0..1` | Dielectric specular reflectance strength (`0.5` ≈ 4 % F0). |
| `--clearcoat` | `0.0` | `0..1` | A second, colorless specular layer (car-paint / lacquer). |
| `--env <FILE>` | — | `.hdr` | Equirectangular HDR environment map reflected by metallic surfaces (see [Environment probe](#hdr-environment-probe)). |
| `--env-intensity` | `1.0` | `≥ 0` | Environment-reflection gain (0 disables the probe reflection). |
| `--env-background` | off | — | Also draw the `--env` probe as the frame's **background sky**, behind every primitive (see [Environment probe](#hdr-environment-probe)). |
| `--env-background-blur` | `0.0` | `0..1` | Blur of the `--env-background` sky (0 = sharp, 1 = fully blurred). |
| `--exposure` | `1.2` | `≥ 0` | Tone-map exposure applied to the linear radiance before the curve. |
| `--ambient` | `0.12` | `≥ 0` | Constant ambient fill (× base color) so shadowed regions aren't pure black. |
| `--tonemap` | `reinhard` | `reinhard`\|`aces` | Tone-map operator (see [Tone mapping](#tone-mapping)). |

## Typed PBR model

The CPU model mirrors the inputs to the shader instead of folding them into one
catch-all struct:

- [`DisneyMaterial`](../crates/trd-core/src/material/disney.rs) owns only
  the **11 Burley 2012 surface parameters**, plus unshaded import metadata.
- [`Lighting`](../crates/trd-core/src/light.rs) owns ambient fill and the
  fixed key/fill/rim rig gain; `Light` and `PointLight` give both uniform arrays
  typed CPU representations. It is a **scene** property — set it on the `Scene`
  you render (`Scene::with_lighting`), not through a renderer setter (#182).
- [`ImageBasedLighting`](../crates/trd-core/src/render/env_map.rs) owns the
  per-object environment-reflection gain; `EnvMapData` owns the HDR probe data
  and `Environment` its GPU binding.
- [`ToneMapping`](../crates/trd-core/src/render/tonemap.rs) owns the per-object
  operator and exposure.

The GPU side is split by **frequency of change** (#182): `PbrSceneUniform`
(group 0, binding 0) carries the camera terms + the light rig and is written
**once per frame**, while `PbrUniform` (group 0, binding 1) is an 80-byte
per-mesh slot a draw selects with a dynamic offset. Both share group 0 because
`pbr.wgsl` already uses all four bind groups and the portable WebGPU baseline
guarantees only four. The single 304-byte uniform this replaces re-encoded the
same rig into every mesh's slot each frame.
The CLI flags drive the common subset, and the rest use neutral defaults:

| Parameter | Default | CLI flag | Meaning |
|---|---|---|---|
| `base_color` | `[1, 1, 1]` | — | Linear-RGB tint multiplied onto the sampled albedo. |
| `metallic` | `0.0` | `--metallic` | Dielectric ↔ metal. |
| `roughness` | `0.5`¹ | `--roughness` | Mirror ↔ fully rough. |
| `specular` | `0.5` | `--specular` | Dielectric specular reflectance. |
| `specular_tint` | `0.0` | — | Tints the dielectric specular toward the base-color hue. |
| `subsurface` | `0.0` | — | Diffuse ↔ subsurface-scattering blend. |
| `anisotropic` | `0.0` | — | Specular anisotropy (0 = isotropic). |
| `sheen` | `0.0` | — | Grazing retro-reflection (e.g. cloth). |
| `sheen_tint` | `0.5` | — | Tints the sheen toward the base-color hue. |
| `clearcoat` | `0.0` | `--clearcoat` | Second colorless specular layer. |
| `clearcoat_gloss` | `1.0` | — | Clearcoat glossiness (0 = satin, 1 = glossy). |

¹ The core default is `0.5`; `trd-cli`'s `--roughness` flag defaults to `0.35`.

`DisneyMaterial::metal()` is a shiny-metal preset (fully metallic and moderately
smooth) — the look used for the drink cans in the Olympic demo.

### Preserved glTF data

`DisneyMaterial::auxiliary` preserves alpha mode/cutoff, opacity, double-sided,
emissive strength, IOR, transmission, and core texture-slot presence. These
fields are deliberately **not consumed by `pbr.wgsl` yet**.
`trd_core::import_gltf_materials` parses glTF material/KHR data from caller-owned
bytes into this model without filesystem access; its per-parameter `sources` map
records whether values came from glTF, an extension-derived mapping, or a
default.

`trd_core::import_glb` additionally imports one triangle primitive, authored
normals, and its embedded base-color / metallic-roughness / normal textures.
`trd-gui` web accepts the GLB directly through
`?mesh=/assets/.../model.glb` and starts it in PBR mode. The glTF path uses
`pbr.wgsl`, the live PBR shader. The original Disney-only `shader/disney.wgsl`
is kept as reference material and is no longer compiled.

## Tone mapping

PBR shading accumulates **linear HDR radiance**, which is tone-mapped to the sRGB
color target:

- **`reinhard`** (default) — per-channel `x / (1 + x)`; trd's historical curve,
  byte-identical to the pre-PBR pipeline.
- **`aces`** — the filmic ACES fit (Narkowicz RRT+ODT, reference
  [`ref/ToneMapping/tonemap.frag`](../ref/ToneMapping/tonemap.frag)); its S-curve
  gives a softer highlight roll-off and retains hue/saturation on bright, strongly
  lit albedo, where per-channel Reinhard desaturates toward grey.

`ToneMapping` remains **per object**, matching the interactive multi-object
viewer. `--exposure` scales radiance before the curve; `Lighting::ambient` adds a
constant fill so shadows aren't crushed to black.

## HDR environment probe

`--env <file.hdr>` binds an **equirectangular HDR** environment map (Radiance
`.hdr`) that metallic surfaces reflect. The image is decoded in the shell (trd-core
does no file I/O) and downscaled to the renderer's 2048 px limit;
`--env-intensity` scales the reflection (0 disables it). Sample maps live in
[`assets/envmap/`](../assets/envmap/) (`uffizi-large.hdr`, `cathedral.hdr`,
`museum.hdr`, `ballroom.hdr`, `grace-new.hdr`).

For glTF PBR, trd precomputes a GGX-filtered PMREM mip chain, diffuse irradiance
map, and split-sum BRDF integration LUT. This avoids view-dependent noise while
preserving roughness-dependent reflections.

The probe can also be **drawn as the background sky**, camera-centered behind
every primitive: `--env-background` (blurred by `--env-background-blur`,
tone-mapped with `--exposure` / `--tonemap`) in `trd-cli` and `trd-app`, the
**Environment background** toggle in `trd-gui`, and `setEnvBackground(enabled,
blur)` in the browser renderers. It is a `RenderOptions` field
(`env_background`) applied by the shared `Scene::from_draws` assembly, so every
front-end gets it from the same inputs (#235 R2) — before that only `trd-gui`
could draw a sky, by reaching around the assembly. The sky needs a bound probe:
with no `--env` the placeholder 1×1 black one is drawn. Its
**yaw is one value**, `Lighting::environment.rotation` (an
[`EnvironmentLight`](../crates/trd-core/src/light.rs)), driving the visible sky
and the reflections together. It used to exist twice, per-mesh and per-scene,
with nothing keeping them equal (#182).

## Interactive editing — `trd-gui`

The interactive viewer starts in PBR mode with `--pbr` and exposes per-object
surface, IBL, and tone-mapping controls (metallic, roughness, clearcoat,
env-intensity, exposure, tone-map), plus Shaded / Roughness / Metallic / Normal
diagnostic views, so you can dial in or inspect a look before baking it into a
`--cli` render:

```sh
cargo run -p trd-gui-app -- --pbr --env assets/envmap/uffizi-large.hdr \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg
```

See also: [`docs/rendering.md`](rendering.md#rendering-appearance) for how PBR sits
alongside the filled / wireframe / textured modes, and the packaged
`examples/olympic-basketball-demo.sh` for a full Disney-PBR + ACES AR demo.
