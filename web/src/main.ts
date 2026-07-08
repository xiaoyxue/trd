// Thin bootstrap wrapper. All rendering logic lives in the Rust/wgpu core
// (trd-core), compiled to wasm. This file must not call the WebGPU API
// directly; from slice 3 it only initialises the wasm module and hands it a
// canvas.

console.log("Hello from trd-web (bun bootstrap).");
