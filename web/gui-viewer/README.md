# trd-gui — browser (wasm) build

The interactive `trd-gui` viewer also targets the browser (issue #97, Slice 4):
the egui UI runs on a `<canvas>` via eframe while `trd-core` renders the scene
**offscreen** (wgpu 30) to RGBA shown as an egui texture (Strategy A). The whole
UI + interaction + rendering are in Rust; `index.html` is the only JS — a thin
bootstrap that loads the wasm module and calls `start(canvas)`.

## Build & run (Bun workspace)

The browser delivery surfaces share the Bun workspace rooted at `web/`. From the
repository root:

```sh
cd web
bun install --frozen-lockfile
bun run --cwd gui-viewer dev  # build wasm + serve on http://localhost:8082
```

`dev` runs `build:wasm` (wasm-pack → `pkg/`) then `serve.ts`, which
bundles `index.html` + `src/main.ts` + the wasm asset **and** statically serves
the repo's real `assets/` directory (so `?mesh=`/`?texture=` fetch the same files
the native viewer reads — no copies). If `pkg/` is already built, `bun run serve`
skips the wasm rebuild. `pkg/`, `dist/`, `node_modules/` are generated (gitignored).

Requires a **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox with WebGPU
enabled) for `trd-core`'s offscreen wgpu renderer — and for egui itself, which
runs on WebGPU through `egui-wgpu` since #257. There is no WebGL fallback.

## Scene URL params (browser equivalents of the native flags)

| Native | Browser URL |
|--------|-------------|
| *(default)* | `http://localhost:8082/` — built-in cube |
| `--mesh assets/meshes/bunny.obj` | `?mesh=/assets/meshes/bunny.obj` |
| `--mesh …bunny.obj --texture …map1.jpg` | `?mesh=/assets/meshes/bunny_with_texture/bunny.obj&texture=/assets/meshes/bunny_with_texture/bunny_uv_map1.jpg` |
| *(browser only)* | `?simple` or `?simple=true` — [simple mode](#simple-mode) |

`?mesh=`/`?texture=` accept any URL the dev server can reach; `/assets/…` maps to
the repo's `assets/` folder. Select **Textured** in the side panel to see a bound
texture.

## Simple mode

For an application that supplies the models itself (#387). Simple mode hides the
**PBR material** section and the **Load model…** / **Delete selected** buttons.
Everything else stays: click-to-pick, **Interaction** (orbit, or rotate /
translate / scale the picked object), **Transform**, **Render mode**,
**Shadows**, **Overlays** and **Selection**. A failed load still appears in the
panel.

The default is the full UI. `?simple` and `?simple=true` turn simple mode on;
no parameter or `?simple=false` keeps the full UI; any other value fails at
startup.

```text
http://localhost:8082/?simple=true&mesh=/assets/meshes/glb/Meshy_AI_Dragon_0804104424_texture.glb&env=/assets/envmap/uffizi-large.hdr
```

From JS/TS, pass `simple` as the last argument to `start`. Load models through
the handle it returns:

```ts
import init, { start } from "./pkg/trd_wasm.js";

await init({ module_or_path: wasmUrl });
const fetchBytes = async (url: string) =>
  new Uint8Array(await (await fetch(url)).arrayBuffer());
const env = await fetchBytes("/assets/envmap/uffizi-large.hdr");
const dragon = await fetchBytes("/models/dragon.glb");

// No picker callback: simple mode has no Load button to call it.
const handle = await start(canvas, [dragon], [], env, undefined, true);

// Later, add another model. Returns void; a failure shows in the viewer panel.
handle.loadModel("chair.glb", await fetchBytes("/models/chair.glb"), env);
```

`serve.ts` also serves [`embed.html`](embed.html) at `/embed`, a working copy of
that pattern. `/embed?case=runtime` starts on the built-in cube and loads the
Dragon two seconds later with no user input. `/embed?case=invalid` loads bytes
that are not a GLB, and the panel must show the error.

## Controls

Same as native: left-drag orbits the camera (or rotates the object, per the side
panel), right/middle-drag moves the object, scroll zooms; the side panel toggles
render mode (Filled/Wireframe/Textured) and overlays.

**Load model** opens the native file chooser for a GLB. When its asynchronous
file read completes, the queued upload wakes the idle GUI automatically; no
additional click is needed. Document-global shadow settings survive model loads
and **Reset view**.
