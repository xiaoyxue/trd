// Thin bootstrap wrapper for the trd renderer.
//
// Its only job is to load the Rust + wgpu core (compiled to wasm) and start it.
// All rendering logic lives in Rust — do NOT call the WebGPU API from here.
import init, { start } from "./pkg/trd.js";

async function main() {
  await init();
  start();
}

main().catch((err) => {
  console.error("Failed to start trd:", err);
  const container = document.getElementById("trd-canvas-container");
  if (container) {
    container.textContent =
      "Failed to start trd. Your browser may not support WebGPU/WebGL2. See the console for details.";
  }
});
