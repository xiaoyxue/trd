// Thin bootstrap wrapper. All rendering logic lives in the Rust/wgpu core
// (trd-core), compiled to wasm and packaged as the "trd-wasm" npm library via
// wasm-pack. This file must not call the WebGPU API directly; it only
// initialises the wasm module and hands it the canvas.
//
// The wasm binary is imported as a bundler asset (`type: "file"`) so bun emits
// and serves it at a URL that works for both the dev server and `bun build`.
// This avoids relying on `import.meta.url`, which the wasm-pack `web` target's
// default init resolves to a path bun does not serve.
import init, { start } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };

const canvas = document.getElementById("trd-canvas");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error('expected a <canvas id="trd-canvas"> element');
}

await init({ module_or_path: wasmUrl });
await start(canvas);
