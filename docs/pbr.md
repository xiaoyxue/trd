# Physically-based rendering (PBR)

`--pbr` shades meshes with the **Disney principled BRDF** (Burley 2012) instead of
flat per-vertex color or a plain textured lookup. The BRDF lives in
[`crates/trd-core/src/disney.wgsl`](../crates/trd-core/src/disney.wgsl) (reference:
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
| `--exposure` | `1.2` | `≥ 0` | Tone-map exposure applied to the linear radiance before the curve. |
| `--ambient` | `0.12` | `≥ 0` | Constant ambient fill (× base color) so shadowed regions aren't pure black. |
| `--tonemap` | `reinhard` | `reinhard`\|`aces` | Tone-map operator (see [Tone mapping](#tone-mapping)). |

## The full material model

The core material ([`trd_core::PbrMaterial`](../crates/trd-core/src/render/pbr.rs))
carries the **complete Burley 2012 parameter set**; the CLI flags above drive the
common subset, and the rest use neutral defaults. Every parameter is in `[0, 1]`
except `base_color` (a linear-RGB tint on the albedo) and the non-negative gains
(`env_intensity`, `exposure`, `ambient`, `light_scale`):

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
| `env_intensity` | `1.0` | `--env-intensity` | Environment-probe reflection gain. |
| `exposure` | `1.2` | `--exposure` | Tone-map exposure. |
| `ambient` | `0.12` | `--ambient` | Constant ambient fill. |
| `light_scale` | `2.5` | — | Scales every virtual light's contribution. |
| `tonemap` | `Reinhard` | `--tonemap` | Tone-map operator. |

¹ The core default is `0.5`; `trd-cli`'s `--roughness` flag defaults to `0.35`.

`PbrMaterial::metal()` is a shiny-metal preset (fully metallic, moderately smooth,
env probe reflected) — the look used for the drink cans in the Olympic demo.

## Tone mapping

PBR shading accumulates **linear HDR radiance**, which is tone-mapped to the sRGB
color target:

- **`reinhard`** (default) — per-channel `x / (1 + x)`; trd's historical curve,
  byte-identical to the pre-PBR pipeline.
- **`aces`** — the filmic ACES fit (Narkowicz RRT+ODT, reference
  [`ref/ToneMapping/tonemap.frag`](../ref/ToneMapping/tonemap.frag)); its S-curve
  gives a softer highlight roll-off and retains hue/saturation on bright, strongly
  lit albedo, where per-channel Reinhard desaturates toward grey.

`--exposure` scales radiance **before** the curve; `--ambient` adds a constant fill
so shadows aren't crushed to black.

## HDR environment probe

`--env <file.hdr>` binds an **equirectangular HDR** environment map (Radiance
`.hdr`) that metallic surfaces reflect. The image is decoded in the shell (trd-core
does no file I/O) and downscaled to the renderer's 2048 px limit;
`--env-intensity` scales the reflection (0 disables it). Sample maps live in
[`assets/envmap/`](../assets/envmap/) (`uffizi-large.hdr`, `cathedral.hdr`,
`museum.hdr`, `ballroom.hdr`, `grace-new.hdr`).

## Interactive editing — `trd-gui`

The interactive viewer starts in PBR mode with `--pbr` and exposes the material +
lighting parameters as **live-editable controls** (metallic, roughness, specular,
clearcoat, env-intensity, exposure, ambient, tone-map), so you can dial in a look
before baking it into a `--cli` render:

```sh
cargo run -p trd-gui -- --pbr --env assets/envmap/uffizi-large.hdr \
  --mesh assets/meshes/bunny_with_texture/bunny.obj \
  --texture assets/meshes/bunny_with_texture/bunny_uv_map1.jpg
```

See also: [`docs/rendering.md`](rendering.md#rendering-appearance) for how PBR sits
alongside the filled / wireframe / textured modes, and the packaged
`examples/olympic-basketball-demo.sh` for a full Disney-PBR + ACES AR demo.
