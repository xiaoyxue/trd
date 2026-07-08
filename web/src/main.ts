// Thin bootstrap wrapper. All rendering logic lives in the Rust/wgpu core
// (trd-core), compiled to wasm via trd-wasm. This file must not call the WebGPU
// API directly; it only initialises the wasm module and hands it a canvas.

import init, { start } from "./generated/trd_wasm.js";

const canvas = document.getElementById("trd-canvas");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error("expected a <canvas id=\"trd-canvas\"> element");
}

await init();
await start(canvas);
