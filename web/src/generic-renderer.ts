// The single, config-driven browser renderer. It is the in-browser twin of
// `render.sh --cli`: `render.sh --web` runs the SAME Arrow producers (mesh +
// texture + params) at the SAME scene flags, writes the resulting stream plus a
// small `config.json` into the served directory, and this module fetches both
// and replays them — to an on-screen `<canvas>` (`--canvas-renderer`, the
// `CanvasRenderer`) or to an offscreen texture painted back to the canvas
// (`--offscreen-renderer`, the `ArrowRenderer`).
//
// The scene (which meshes/texture, camera, draws, wireframe/textured/aabb/axes/
// axes-local, background compositing) is fixed by render.sh at generation time,
// exactly like the CLI. Only playback `?fps=N` is a live URL param (the render
// resolution is baked into the stream, so it is a render.sh argument).
import init, { ArrowRenderer, CanvasRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };

/// The flags render.sh bakes alongside the generated `stream.arrow`. Mirrors the
/// `--cli` scene flags plus the chosen web target and default playback rate.
interface RenderConfig {
  /// `canvas` = on-screen `CanvasRenderer`; `offscreen` = `ArrowRenderer`
  /// rendering to a texture then painted to a 2D canvas.
  target: "canvas" | "offscreen";
  /// Base render mode for meshes without a per-draw override.
  mode: "filled" | "wireframe" | "textured";
  showAabb: boolean;
  showAxes: boolean;
  showLocalAxes: boolean;
  /// Composite each frame's `frame_ref` still beneath the scene (#63).
  background: boolean;
  /// Baked render resolution (matches the stream's CV `k`, so it is fixed).
  width: number;
  height: number;
  /// Default playback rate; `?fps=N` overrides it live.
  fps: number;
}

const STATUS = document.getElementById("trd-status");

function setStatus(message: string): void {
  if (STATUS) {
    STATUS.textContent = message;
  }
}

function fail(message: string): never {
  setStatus(`error: ${message}`);
  throw new Error(message);
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) {
    fail(`failed to fetch ${url}: ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

/// Fetches + decodes a background still to tightly-packed RGBA, rescaled to the
/// render resolution so the frame-plane texture allocates once (FrameFit stretch
/// handles any aspect). Decoded stills are cached by URL.
const backgroundCache = new Map<string, Uint8Array>();

async function decodeBackground(url: string, width: number, height: number): Promise<Uint8Array> {
  const cached = backgroundCache.get(url);
  if (cached) {
    return cached;
  }
  const response = await fetch(url);
  if (!response.ok) {
    fail(`failed to load background ${url}: ${response.status}`);
  }
  // Decode AND downscale to the render resolution in one step: `resizeWidth/
  // Height` lets the browser decode straight to the target size (a much cheaper
  // path than decoding the full-resolution still then scaling it), so the first
  // playback pass — which populates the cache — does not stutter on large stills.
  const bitmap = await createImageBitmap(await response.blob(), {
    resizeWidth: width,
    resizeHeight: height,
    resizeQuality: "high",
  });
  const offscreen = new OffscreenCanvas(width, height);
  const context = offscreen.getContext("2d");
  if (!context) {
    fail("failed to acquire OffscreenCanvas 2D context");
  }
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = new Uint8Array(context.getImageData(0, 0, width, height).data.buffer.slice(0));
  backgroundCache.set(url, rgba);
  return rgba;
}

function resolveFps(configFps: number): number {
  const query = new URLSearchParams(window.location.search);
  const requested = Number(query.get("fps"));
  if (Number.isFinite(requested) && requested >= 1 && requested <= 240) {
    return requested;
  }
  return configFps >= 1 && configFps <= 240 ? configFps : 24;
}

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

async function main(): Promise<void> {
  setStatus("loading config…");
  const configResponse = await fetch("./config.json");
  if (!configResponse.ok) {
    fail(
      `no config.json (${configResponse.status}) — run \`render.sh --web\` to generate the stream`,
    );
  }
  const config = (await configResponse.json()) as RenderConfig;

  setStatus("loading stream…");
  const stream = await fetchBytes("./stream.arrow");

  await init({ module_or_path: wasmUrl });

  const canvas = document.getElementById("trd-canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    fail("missing #trd-canvas element");
  }
  canvas.width = config.width;
  canvas.height = config.height;

  const fps = resolveFps(config.fps);

  if (config.target === "offscreen") {
    await runOffscreen(canvas, config, stream, fps);
  } else {
    await runCanvas(canvas, config, stream, fps);
  }
}

async function runCanvas(
  canvas: HTMLCanvasElement,
  config: RenderConfig,
  stream: Uint8Array,
  fps: number,
): Promise<void> {
  const renderer = await CanvasRenderer.create(canvas);
  applyMode(renderer, config);
  const total = renderer.loadIpc(stream);
  if (total === 0) {
    fail("stream carried no frames");
  }

  const interval = 1000 / fps;
  let index = 0;
  setStatus(`canvas — ${total} frames @ ${fps}fps (${config.width}×${config.height})`);
  for (;;) {
    if (config.background) {
      await uploadBackground(renderer, renderer.frameRef(index), config);
    }
    renderer.renderIndex(index);
    index = (index + 1) % total;
    await sleep(interval);
  }
}

async function runOffscreen(
  canvas: HTMLCanvasElement,
  config: RenderConfig,
  stream: Uint8Array,
  fps: number,
): Promise<void> {
  const context = canvas.getContext("2d");
  if (!context) {
    fail("failed to acquire 2D context for the offscreen display");
  }
  const renderer = await ArrowRenderer.create(config.width, config.height);
  applyMode(renderer, config);
  const total = renderer.loadIpc(stream);
  if (total === 0) {
    fail("stream carried no frames");
  }

  const interval = 1000 / fps;
  let index = 0;
  setStatus(`offscreen — ${total} frames @ ${fps}fps (${config.width}×${config.height})`);
  for (;;) {
    if (config.background) {
      await uploadBackground(renderer, renderer.frameRef(index), config);
    }
    const rgba = await renderer.renderIndex(index);
    const image = new ImageData(new Uint8ClampedArray(rgba), config.width, config.height);
    context.putImageData(image, 0, 0);
    index = (index + 1) % total;
    await sleep(interval);
  }
}

/// The scene-mode + overlay flags are identical across both renderers; applied
/// once before playback. `setTextured`/`setWireframe` are mutually exclusive; the
/// default (filled) leaves per-vertex color.
function applyMode(renderer: CanvasRenderer | ArrowRenderer, config: RenderConfig): void {
  if (config.mode === "wireframe") {
    renderer.setWireframe(true);
  } else if (config.mode === "textured") {
    renderer.setTextured(true);
  }
  renderer.setShowAabb(config.showAabb);
  renderer.setShowAxes(config.showAxes);
  renderer.setShowLocalAxes(config.showLocalAxes);
  if (config.background) {
    renderer.setCompositeFrame(true);
  }
}

/// Resolves a frame's background reference to RGBA and uploads it as the reused
/// frame-plane texture. A frame with no reference keeps the previous background.
async function uploadBackground(
  renderer: CanvasRenderer | ArrowRenderer,
  ref: string | undefined,
  config: RenderConfig,
): Promise<void> {
  if (!ref) {
    return;
  }
  const rgba = await decodeBackground(`./${ref}`, config.width, config.height);
  renderer.updateFrameTextureRgba(rgba, config.width, config.height);
}

await main();
