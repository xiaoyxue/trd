// Thin bootstrap for the browser `trd-gui` viewer (issue #97, Slice 4). The whole
// UI + interaction + offscreen rendering live in Rust (the trd-gui wasm module);
// this file only loads that module and hands it the canvas — the browser twin of
// `main.rs`'s `eframe::run_native`. Bundled/served by the shared Bun workspace.
import init, { start } from "../pkg/trd_gui.js";
import wasmUrl from "../pkg/trd_gui_bg.wasm" with { type: "file" };

async function main(): Promise<void> {
  // wasm-bindgen's default wasm path breaks once bundled, so pass the bundler's
  // asset URL explicitly (same pattern as web/viewer/src/viewer.ts).
  await init({ module_or_path: wasmUrl });

  const canvas = document.getElementById("trd-gui-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("missing #trd-gui-canvas");
  }

  // `?mesh=<url>` (repeatable) / `?texture=<url>` / `?env=<url>` are the browser
  // equivalents of the native `--mesh` / `--texture` / `--env` flags: fetch the
  // OBJ/GLB bytes / image bytes / HDR probe bytes and hand them to Rust. Multiple `?mesh=`
  // params load multiple objects (laid out side-by-side; click to select one).
  // Absent → built-in cube / no texture / no env probe. `?env=` starts PBR mode.
  const params = new URLSearchParams(location.search);

  const meshUrls = params.getAll("mesh");
  const meshBytes: Uint8Array[] = await Promise.all(
    meshUrls.map(async (url) => {
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`failed to fetch mesh "${url}": ${res.status} ${res.statusText}`);
      }
      return new Uint8Array(await res.arrayBuffer());
    }),
  );

  // `?texture=<url>` is **positional**: the i-th `?texture=` skins the i-th
  // `?mesh=` (each object its own diffuse). A missing slot → an empty array →
  // untextured (1×1 white). An empty string entry also means "no texture".
  const textureUrls = params.getAll("texture");
  const textureBytes: Uint8Array[] = await Promise.all(
    meshUrls.map(async (_mesh, i) => {
      const url = textureUrls[i];
      if (!url) {
        return new Uint8Array();
      }
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`failed to fetch texture "${url}": ${res.status} ${res.statusText}`);
      }
      return new Uint8Array(await res.arrayBuffer());
    }),
  );

  const envUrl = params.get("env");
  let envBytes: Uint8Array | undefined;
  if (envUrl) {
    const res = await fetch(envUrl);
    if (!res.ok) {
      throw new Error(`failed to fetch env map "${envUrl}": ${res.status} ${res.statusText}`);
    }
    envBytes = new Uint8Array(await res.arrayBuffer());
  }

  await start(canvas, meshBytes, textureBytes, envBytes);
}

main().catch((err) => {
  console.error("trd-gui failed to start:", err);
  document.body.innerHTML =
    `<pre style="color:#f88;padding:1rem;font-family:monospace">trd-gui failed to start:\n${err}\n\n` +
    "A WebGPU-capable browser is required (Chrome/Edge 113+, or Firefox with WebGPU enabled).</pre>";
});
