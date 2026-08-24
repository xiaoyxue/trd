// Thin bootstrap for the browser `trd-gui` viewer (issue #97, Slice 4). The whole
// UI + interaction + offscreen rendering live in Rust (the trd-gui wasm module);
// this file only loads that module and hands it the canvas — the browser twin of
// `main.rs`'s `eframe::run_native`. Bundled/served by the shared Bun workspace.
// The probe a runtime-loaded model is lit by when the viewer was opened without
// `?env=`. Bundled (not fetched by path) so the built `dist/` is self-contained,
// and it is the same Uffizi probe the video editor lights the Dragon with.
import uffiziEnvUrl from "../../../assets/envmap/uffizi-large.hdr" with { type: "file" };
import init, { type GuiHandle, start } from "../pkg/trd_wasm.js";
import wasmUrl from "../pkg/trd_wasm_bg.wasm" with { type: "file" };

/// Fetched at most once, and only if a load actually needs it.
let defaultEnvBytes: Promise<Uint8Array> | undefined;
function uffiziProbe(): Promise<Uint8Array> {
  defaultEnvBytes ??= fetch(uffiziEnvUrl).then(async (response) => {
    if (!response.ok) {
      throw new Error(`failed to fetch the Uffizi probe: ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  });
  return defaultEnvBytes;
}

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
  // The viewer is lit by image-based lighting alone, so a probe is not optional:
  // without `?env=` it falls back to the built-in Uffizi one rather than
  // rendering an unlit scene.
  const envBytes: Uint8Array = envUrl
    ? await fetch(envUrl).then(async (res) => {
        if (!res.ok) {
          throw new Error(`failed to fetch env map "${envUrl}": ${res.status} ${res.statusText}`);
        }
        return new Uint8Array(await res.arrayBuffer());
      })
    : await uffiziProbe();

  // The "Load model…" button lives in the Rust panel, but opening a file picker
  // needs a user gesture the browser only grants the page — so Rust calls out to
  // this hidden input and the bytes go back in through the handle. JS parses
  // nothing: `accept` is a hint to the dialog, and Rust re-checks the magic.
  const picker = document.createElement("input");
  picker.type = "file";
  picker.accept = ".glb,model/gltf-binary";
  picker.hidden = true;
  document.body.appendChild(picker);

  let handle: GuiHandle | undefined;
  picker.addEventListener("change", () => {
    const file = picker.files?.[0];
    // Reset so picking the same file twice still fires a change event.
    picker.value = "";
    if (!file || !handle) {
      return;
    }
    void (async () => {
      const bytes = new Uint8Array(await file.arrayBuffer());
      handle?.loadModel(file.name, bytes, await uffiziProbe());
    })();
  });

  handle = await start(canvas, meshBytes, textureBytes, envBytes, () => picker.click());
}

main().catch((err) => {
  console.error("trd-gui failed to start:", err);
  document.body.innerHTML =
    `<pre style="color:#f88;padding:1rem;font-family:monospace">trd-gui failed to start:\n${err}\n\n` +
    "A WebGPU-capable browser is required (Chrome/Edge 113+, or Firefox with WebGPU enabled).</pre>";
});
