// Thin bootstrap for the browser `trd-gui` viewer (issue #97, Slice 4). The whole
// UI + interaction + offscreen rendering live in Rust (the trd-gui wasm module);
// this file only loads that module and hands it the canvas — the browser twin of
// `main.rs`'s `eframe::run_native`. Bundled/served by bun, mirroring `web/`.
import init, { start } from "../pkg/trd_gui.js";
import wasmUrl from "../pkg/trd_gui_bg.wasm" with { type: "file" };

async function main(): Promise<void> {
  // wasm-bindgen's default wasm path breaks once bundled, so pass the bundler's
  // asset URL explicitly (same pattern as web/src/viewer.ts).
  await init({ module_or_path: wasmUrl });

  const canvas = document.getElementById("trd-gui-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("missing #trd-gui-canvas");
  }

  // `?mesh=<url>` / `?texture=<url>` are the browser equivalents of the native
  // `--mesh` / `--texture` flags: fetch the OBJ text / image bytes and hand them
  // to Rust (`Mesh::from_obj` / `decode_texture`). Absent → built-in cube / no
  // texture.
  const params = new URLSearchParams(location.search);

  const meshUrl = params.get("mesh");
  let meshObj: string | undefined;
  if (meshUrl) {
    const res = await fetch(meshUrl);
    if (!res.ok) {
      throw new Error(`failed to fetch mesh "${meshUrl}": ${res.status} ${res.statusText}`);
    }
    meshObj = await res.text();
  }

  const textureUrl = params.get("texture");
  let textureBytes: Uint8Array | undefined;
  if (textureUrl) {
    const res = await fetch(textureUrl);
    if (!res.ok) {
      throw new Error(`failed to fetch texture "${textureUrl}": ${res.status} ${res.statusText}`);
    }
    textureBytes = new Uint8Array(await res.arrayBuffer());
  }

  // `?backend=arrow` is the browser equivalent of native `--backend arrow`
  // (Arrow wire round-trip); absent → the direct in-process render.
  const backend = params.get("backend") ?? undefined;

  await start(canvas, meshObj, textureBytes, backend);
}

main().catch((err) => {
  console.error("trd-gui failed to start:", err);
  document.body.innerHTML =
    `<pre style="color:#f88;padding:1rem;font-family:monospace">trd-gui failed to start:\n${err}\n\n` +
    "A WebGPU-capable browser is required (Chrome/Edge 113+, or Firefox with WebGPU enabled).</pre>";
});
