// Thin bootstrap wrapper. All rendering logic lives in the Rust/wgpu core
// (trd-core), compiled to wasm and packaged as the "trd-wasm" npm library via
// wasm-pack. This file must not call the WebGPU API directly; it only
// initialises the wasm module and hands it the canvas.
import init, { start } from "trd-wasm";

const canvas = document.getElementById("trd-canvas");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error("expected a <canvas id=\"trd-canvas\"> element");
}

await init();
await start(canvas);
