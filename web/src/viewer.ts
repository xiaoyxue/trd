// The single, config-driven browser viewer. It is the in-browser twin of
// `render.sh --cli`: `render.sh --web` runs the SAME Arrow producers (mesh +
// texture + params) at the SAME scene flags, writes the resulting stream plus a
// small `config.json` into the served directory, and this module fetches both
// and replays them — to an on-screen `<canvas>` (`--canvas-renderer`, the
// `CanvasRenderer`) or to an offscreen texture painted back to the canvas
// (`--offscreen-renderer`, the `OffscreenRenderer`).
//
// The scene (which meshes/texture, camera, draws, wireframe/textured/aabb/axes/
// axes-local, background compositing) is fixed by render.sh at generation time,
// exactly like the CLI. Only playback `?fps=N` is a live URL param (the render
// resolution is baked into the stream, so it is a render.sh argument).
import init, { CanvasRenderer, OffscreenRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };

/// The flags render.sh bakes alongside the generated `stream.arrow`. Mirrors the
/// `--cli` scene flags plus the chosen web target and default playback rate.
interface RenderConfig {
  /// `canvas` = on-screen `CanvasRenderer`; `offscreen` = `OffscreenRenderer`
  /// rendering to a texture then painted to a 2D canvas.
  target: "canvas" | "offscreen";
  /// Base render mode for meshes without a per-draw override. `pbr` shades the
  /// bound albedo with the Disney principled BRDF (see `pbr`/`env`).
  mode: "filled" | "wireframe" | "textured" | "pbr";
  /// Disney PBR material (present iff `mode === "pbr"`) — the browser twin of
  /// trd-cli's `--metallic/--roughness/…` flags, forwarded verbatim so the web
  /// render matches `trd-app --pbr`.
  pbr?: {
    metallic: number;
    roughness: number;
    specular: number;
    clearcoat: number;
    envIntensity: number;
    exposure: number;
    ambient: number;
    tonemap: "reinhard" | "aces";
  };
  /// Equirectangular Radiance `.hdr` env probe URL (relative to the served
  /// root), reflected by metallic PBR surfaces. Decoded in-wasm; only used when
  /// `mode === "pbr"`.
  env?: string;
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

/// Pre-decodes every distinct background still into the cache before playback
/// starts, so the first loop does not stutter on per-frame JPEG decode (the
/// browser twin of trd-app buffering the whole stream up front). Decoded
/// concurrently in small batches; a status line reports progress.
async function preloadBackgrounds(
  renderer: CanvasRenderer | OffscreenRenderer,
  total: number,
  config: RenderConfig,
): Promise<void> {
  if (!config.background) {
    return;
  }
  const refs: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < total; i++) {
    const ref = renderer.frameRef(i);
    if (ref && !seen.has(ref)) {
      seen.add(ref);
      refs.push(ref);
    }
  }
  const batch = 8;
  for (let start = 0; start < refs.length; start += batch) {
    const slice = refs.slice(start, start + batch);
    await Promise.all(
      slice.map((ref) => decodeBackground(`./${ref}`, config.width, config.height)),
    );
    setStatus(`decoding backgrounds… ${Math.min(start + batch, refs.length)}/${refs.length}`);
  }
}

async function runCanvas(
  canvas: HTMLCanvasElement,
  config: RenderConfig,
  stream: Uint8Array,
  fps: number,
): Promise<void> {
  const renderer = await CanvasRenderer.create(canvas);
  await applyMode(renderer, config);
  const total = renderer.loadIpc(stream);
  if (total === 0) {
    fail("stream carried no frames");
  }
  await preloadBackgrounds(renderer, total, config);

  setStatus(`canvas — ${total} frames @ ${fps}fps (${config.width}×${config.height})`);
  // Present on the display's vsync via requestAnimationFrame, but choose WHICH
  // frame to show from wall-clock elapsed time, so playback runs at exactly `fps`
  // regardless of the monitor refresh rate (the browser twin of trd-app's
  // wall-clock advance()). rAF is capped at the refresh rate, so a slower `fps`
  // simply repeats the same frame across several callbacks; we re-render only
  // when the selected frame index changes.
  const start = performance.now();
  let shown = -1;
  const tick = (now: number): void => {
    requestAnimationFrame(tick);
    const index = Math.floor(((now - start) / 1000) * fps) % total;
    if (index === shown) {
      return;
    }
    shown = index;
    if (config.background) {
      uploadCachedBackground(renderer, renderer.frameRef(index), config);
    }
    renderer.renderIndex(index);
  };
  requestAnimationFrame(tick);
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
  const renderer = await OffscreenRenderer.create(config.width, config.height);
  await applyMode(renderer, config);
  const total = renderer.loadIpc(stream);
  if (total === 0) {
    fail("stream carried no frames");
  }
  await preloadBackgrounds(renderer, total, config);

  setStatus(`offscreen — ${total} frames @ ${fps}fps (${config.width}×${config.height})`);
  // Same wall-clock frame selection as the canvas path, scheduled on rAF. Here
  // `renderIndex` is async (offscreen texture → RGBA readback), so a `busy` guard
  // skips ticks while a readback is still in flight rather than overlapping them.
  const start = performance.now();
  let shown = -1;
  let busy = false;
  const tick = (now: number): void => {
    requestAnimationFrame(tick);
    if (busy) {
      return;
    }
    const index = Math.floor(((now - start) / 1000) * fps) % total;
    if (index === shown) {
      return;
    }
    shown = index;
    busy = true;
    void (async () => {
      if (config.background) {
        uploadCachedBackground(renderer, renderer.frameRef(index), config);
      }
      const rgba = await renderer.renderIndex(index);
      context.putImageData(
        new ImageData(new Uint8ClampedArray(rgba), config.width, config.height),
        0,
        0,
      );
      busy = false;
    })();
  };
  requestAnimationFrame(tick);
}

/// The scene-mode + overlay flags are identical across both renderers; applied
/// once before playback. `setTextured`/`setWireframe`/`setPbr` are mutually
/// exclusive; the default (filled) leaves per-vertex color. For `pbr` this also
/// forwards the Disney material and fetches + decodes the HDR env probe (async),
/// so the browser matches `trd-app --pbr` byte-for-byte.
async function applyMode(
  renderer: CanvasRenderer | OffscreenRenderer,
  config: RenderConfig,
): Promise<void> {
  if (config.mode === "wireframe") {
    renderer.setWireframe(true);
  } else if (config.mode === "textured") {
    renderer.setTextured(true);
  } else if (config.mode === "pbr" && config.pbr) {
    const pbr = config.pbr;
    renderer.setPbr(true);
    renderer.setPbrMaterial(
      pbr.metallic,
      pbr.roughness,
      pbr.specular,
      pbr.clearcoat,
      pbr.envIntensity,
      pbr.exposure,
      pbr.ambient,
      pbr.tonemap,
    );
    if (config.env) {
      setStatus("loading environment map…");
      renderer.setEnvMapHdr(await fetchBytes(config.env));
    }
  }
  renderer.setShowAabb(config.showAabb);
  renderer.setShowAxes(config.showAxes);
  renderer.setShowLocalAxes(config.showLocalAxes);
  if (config.background) {
    renderer.setCompositeFrame(true);
  }
}

/// Uploads a frame's background from the preloaded cache as the reused
/// frame-plane texture. Synchronous (no decode) so it can run inside the rAF
/// callback; `preloadBackgrounds` guarantees the cache hit. A frame with no
/// reference — or one not yet cached — keeps the previous background.
function uploadCachedBackground(
  renderer: CanvasRenderer | OffscreenRenderer,
  ref: string | undefined,
  config: RenderConfig,
): void {
  if (!ref) {
    return;
  }
  const rgba = backgroundCache.get(`./${ref}`);
  if (rgba) {
    renderer.updateFrameTextureRgba(rgba, config.width, config.height);
  }
}

await main();
