// An embedding application in miniature (#387), served at `/embed`: it starts
// the viewer in simple mode, keeps the returned handle, and loads GLBs through
// it — the path simple mode leaves once the Load button is gone.
//
//   /embed?case=runtime  load a valid GLB after startup, with no user input
//   /embed?case=invalid  load bytes that are not a GLB; the panel must say so
import uffiziEnvUrl from "../../../assets/envmap/uffizi-large.hdr" with { type: "file" };
import init, { start } from "../pkg/trd_wasm.js";
import wasmUrl from "../pkg/trd_wasm_bg.wasm" with { type: "file" };

const DRAGON_URL = "/assets/meshes/glb/Meshy_AI_Dragon_0804104424_texture.glb";
// Long enough that the viewer has settled into idle, so the load has to wake it.
const LOAD_DELAY_MS = 2000;

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch "${url}": ${response.status} ${response.statusText}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function main(): Promise<void> {
  const testCase = new URLSearchParams(location.search).get("case");
  if (testCase !== "runtime" && testCase !== "invalid") {
    throw new Error(`?case must be "runtime" or "invalid", got "${testCase}"`);
  }

  await init({ module_or_path: wasmUrl });
  const canvas = document.getElementById("trd-gui-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("missing #trd-gui-canvas");
  }
  const envBytes = await fetchBytes(uffiziEnvUrl);

  // No meshes: the viewer starts with its built-in cube, so a successful
  // runtime load is visibly a second object.
  const handle = await start(canvas, [], [], envBytes, undefined, true);

  const bytes =
    testCase === "runtime"
      ? await fetchBytes(DRAGON_URL)
      : new TextEncoder().encode("this is not a GLB");
  const name = testCase === "runtime" ? "dragon.glb" : "broken.glb";
  setTimeout(() => handle.loadModel(name, bytes, envBytes), LOAD_DELAY_MS);
}

main().catch((err) => {
  console.error("trd-gui embed failed to start:", err);
  document.body.innerHTML = `<pre style="color:#f88;padding:1rem;font-family:monospace">trd-gui embed failed to start:\n${err}</pre>`;
});
