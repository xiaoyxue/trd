# trd-gui — browser (wasm) build

The interactive `trd-gui` viewer also targets the browser (issue #97, Slice 4):
the egui UI runs on a `<canvas>` via eframe while `trd-core` renders the scene
**offscreen** (wgpu 30) to RGBA shown as an egui texture (Strategy A). The whole
UI + interaction + rendering are in Rust; `index.html` is the only JS — a thin
bootstrap that loads the wasm module and calls `start(canvas)`.

## Build & run (bun)

Served/bundled with **bun**, mirroring the repo's `web/` folder. From this
directory (`crates/trd-gui/web/`):

```sh
bun install          # once, for the biome/tsc dev tools
bun run dev          # build the wasm (wasm-pack) + serve on http://localhost:8080
```

`bun run dev` runs `build:wasm` (wasm-pack → `pkg/`) then `serve.ts`, which
bundles `index.html` + `src/main.ts` + the wasm asset **and** statically serves
the repo's real `assets/` directory (so `?mesh=`/`?texture=` fetch the same files
the native viewer reads — no copies). If `pkg/` is already built, `bun run serve`
skips the wasm rebuild. `pkg/`, `dist/`, `node_modules/` are generated (gitignored).

Requires a **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox with WebGPU
enabled) for `trd-core`'s offscreen wgpu renderer.

## Scene URL params (browser equivalents of the native flags)

| Native | Browser URL |
|--------|-------------|
| *(default)* | `http://localhost:8080/` — built-in cube |
| `--mesh assets/meshes/bunny.obj` | `?mesh=/assets/meshes/bunny.obj` |
| `--mesh …bunny.obj --texture …map1.jpg` | `?mesh=/assets/meshes/bunny_with_texture/bunny.obj&texture=/assets/meshes/bunny_with_texture/bunny_uv_map1.jpg` |
| `--backend arrow` | append `&backend=arrow` (the Arrow wire round-trip) |

`?mesh=`/`?texture=` accept any URL the dev server can reach; `/assets/…` maps to
the repo's `assets/` folder. Select **Textured** in the side panel to see a bound
texture.

## Controls

Same as native: left-drag orbits the camera (or rotates the object, per the side
panel), right/middle-drag moves the object, scroll zooms; the side panel toggles
render mode (Filled/Wireframe/Textured) and overlays.
