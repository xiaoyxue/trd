import { ArrowSceneDocument, VideoEditingHandle } from "../pkg/trd_wasm.js";
import "../src/main.ts";

// Observe real host/WASM calls; do not replace rendering, media or the GUI controls.
let handle: VideoEditingHandle | undefined;
const calls: Record<string, number> = {};
const media = { playing: false, loaded: false, frame: -1, seconds: -1, sourceChanges: 0 };
const status = VideoEditingHandle.prototype.setVideoStatus;
VideoEditingHandle.prototype.setVideoStatus = function (loaded, playing) {
  handle = this;
  media.loaded = loaded;
  media.playing = playing;
  return status.call(this, loaded, playing);
};
const present = VideoEditingHandle.prototype.presentVideoFrame;
VideoEditingHandle.prototype.presentVideoFrame = function (frame, index, seconds, duration) {
  media.frame = index;
  media.seconds = seconds;
  return present.call(this, frame, index, seconds, duration);
};
const source = VideoEditingHandle.prototype.setVideoSourceInfo;
VideoEditingHandle.prototype.setVideoSourceInfo = function (kind, name, size) {
  media.sourceChanges++;
  return source.call(this, kind, name, size);
};
for (const name of ["loadArrow", "resetState", "exportArrow", "seekToSeconds"] as const) {
  const method = VideoEditingHandle.prototype[name];
  // Reflect keeps the generated ABI's distinct signatures; only counts are added.
  Object.defineProperty(VideoEditingHandle.prototype, name, {
    configurable: true,
    value: function (this: VideoEditingHandle, ...args: unknown[]) {
      handle = this;
      calls[name] = (calls[name] ?? 0) + 1;
      return Reflect.apply(method, this, args);
    },
  });
}

function assert(value: unknown, message: string): asserts value {
  if (!value) throw new Error(message);
}
async function bytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  assert(response.ok, `fetch ${url}: ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
async function hash(value: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", Uint8Array.from(value).buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
async function rejected(work: () => Promise<unknown>, pattern: RegExp): Promise<void> {
  let message = "";
  try {
    await work();
  } catch (error) {
    message = String(error);
  }
  assert(pattern.test(message), `expected ${pattern}, got ${message || "success"}`);
}
async function waitFor(work: () => boolean, label: string): Promise<void> {
  const start = performance.now();
  while (!work()) {
    assert(performance.now() - start < 10_000, `timed out: ${label}`);
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

const report: Array<Record<string, unknown>> = [];
async function run() {
  const api = await window.trdVideoEditorReady;
  assert(handle, "real WASM handle was observed");
  const wasm = handle;
  const sources = await Promise.all(
    ["params", "single", "multiple"].map((name) => bytes(`/input/${name}.arrow`)),
  );
  const exports: Record<
    string,
    { bytes: number; sha256: string; rows: number; meshIds: string[] }
  > = {};
  for (const [index, name] of ["params", "single", "multiple"].entries()) {
    const input = sources[index];
    assert(input, `missing ${name}`);
    await wasm.loadArrow(input);
    await wasm.seekToSeconds(4.5);
    assert(media.frame === 108, `raw WASM seek did not present 108: ${media.frame}`);
    const raw = await wasm.exportArrow();
    await api.resetState();
    await rejected(() => wasm.exportArrow(), /no current Arrow/);
    const cut = Math.floor(input.length / 2);
    await api.loadArrow([input.slice(0, cut), input.slice(cut)]);
    await api.seekToSeconds(4.5);
    const adapted = await api.exportArrow();
    assert((await hash(raw)) === (await hash(adapted)), `${name}: raw WASM and page API differ`);
    const sourceDocument = ArrowSceneDocument.fromArrow(input);
    try {
      assert(
        (await hash(sourceDocument.exportArrow())) === (await hash(adapted)),
        `${name}: source changed on load/export`,
      );
      exports[name] = {
        bytes: adapted.length,
        sha256: await hash(adapted),
        rows: sourceDocument.frameCount(),
        meshIds: sourceDocument.meshIds(),
      };
    } finally {
      sourceDocument.free();
    }
    report.push({ name, frame: media.frame, bytes: adapted.length });
  }
  const single = sources[1];
  assert(single, "single fixture missing");
  const current = await api.exportArrow();
  const old = single.slice();
  const version = new TextEncoder().encode("0.0.7");
  let replaced = false;
  for (let i = 0; i <= old.length - version.length; i++) {
    if (version.every((value, offset) => old[i + offset] === value)) {
      old[i + 4] = 54;
      replaced = true;
      break;
    }
  }
  assert(replaced, "0.0.7 input metadata missing");
  await rejected(() => wasm.loadArrow(old), /0\.0\.6/);
  await rejected(() => wasm.loadArrow(new Uint8Array()), /.+/);
  assert(
    (await hash(current)) === (await hash(await wasm.exportArrow())),
    "failed load changed the scene",
  );

  const document = ArrowSceneDocument.fromArrow(single);
  try {
    for (let row = 0; row < document.frameCount(); row++) {
      const matrix = document.getModel(row, 0);
      const x = matrix[12];
      assert(x !== undefined, "model matrix must have 16 elements");
      matrix[12] = x + 0.2;
      document.setModel(row, 0, matrix);
    }
    const edited = document.exportArrow();
    await wasm.loadArrow(edited);
    await api.seekToSeconds(2);
    const before = await api.exportArrow();
    await api.resetState();
    await api.loadArrow(before);
    for (const second of [4.5, 221 / 24, 222 / 24, 287 / 24, 0]) {
      await wasm.seekToSeconds(second);
      assert(media.frame === Math.round(second * 24), `wrong frame at ${second}: ${media.frame}`);
    }
    assert(
      (await hash(before)) === (await hash(await wasm.exportArrow())),
      "replay changed edited matrices",
    );
    exports.edited = {
      bytes: before.length,
      sha256: await hash(before),
      rows: document.frameCount(),
      meshIds: document.meshIds(),
    };
  } finally {
    document.free();
  }
  await Promise.all([wasm.seekToSeconds(2), wasm.seekToSeconds(3), wasm.seekToSeconds(4)]);
  assert(
    media.frame === 96,
    `overlapping raw WASM seeks did not settle to final frame: ${media.frame}`,
  );
  await rejected(() => wasm.seekToSeconds(Number.NaN), /finite/);
  const a = api.resetState();
  const b = api.loadArrow(single);
  const c = api.exportArrow();
  await Promise.all([a, b]);
  assert((await c).length > 0, "queued reset/load/export failed");
  await api.seekToSeconds(1);
  report.push({ name: "lifecycle", calls: { ...calls }, sourceChanges: media.sourceChanges });
  return { report, exports, calls, media };
}

async function resetPlaying() {
  const api = await window.trdVideoEditorReady;
  assert(handle, "WASM handle missing");
  assert(media.playing, "start playback through the actual GUI before this case");
  await api.seekToSeconds(2);
  const before = { ...media };
  await handle.resetState();
  assert(media.playing && media.loaded, "reset stopped the video");
  assert(media.sourceChanges === before.sourceChanges, "reset reopened the video");
  assert(media.seconds >= before.seconds, "reset rewound the video");
  await waitFor(() => media.seconds > before.seconds + 0.25, "video continues after reset");
  await api.loadArrow(await bytes("/input/single.arrow"));
  assert(media.playing, "loading new Arrow paused the video");
  const snapshot = await api.exportArrow();
  report.push({ name: "playing-reset-load", before, after: { ...media }, bytes: snapshot.length });
  return report.at(-1);
}

Object.assign(window, {
  trdApiE2e: { run, resetPlaying, state: () => ({ calls: { ...calls }, media: { ...media } }) },
});
