# trd-gui — browser (wasm) build

The interactive `trd-gui` viewer also targets the browser (issue #97, Slice 4):
the egui UI runs on a `<canvas>` via eframe while `trd-core` renders the scene
**offscreen** (wgpu 30) to RGBA shown as an egui texture (Strategy A). The whole
UI + interaction + rendering are in Rust; `index.html` is the only JS — a thin
bootstrap that loads the wasm module and calls `start(canvas)`.

## Build

From the repo root, build the wasm-bindgen web package into `pkg/`:

```sh
wasm-pack build crates/trd-gui --target web --out-dir web/pkg
```

or, equivalently, with `cargo` + `wasm-bindgen-cli`:

```sh
cargo build -p trd-gui --release --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir crates/trd-gui/web/pkg --out-name trd_gui \
  target/wasm32-unknown-unknown/release/trd_gui.wasm
```

`crates/trd-gui/web/pkg/` is generated (gitignored).

## Run

Serve `crates/trd-gui/web/` over HTTP (ES modules + wasm need a server, not
`file://`) with any static server, e.g.:

```sh
python3 -m http.server 8080 --directory crates/trd-gui/web
# then open http://localhost:8080
```

Requires a **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox with WebGPU
enabled) for `trd-core`'s offscreen wgpu renderer.

## Controls

Same as native: left-drag orbits the camera (or rotates the object, per the side
panel), right/middle-drag moves the object, scroll zooms; the side panel toggles
render mode (Filled/Wireframe/Textured) and overlays.
