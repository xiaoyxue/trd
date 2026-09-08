import { describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  Field,
  FixedSizeList,
  Float32,
  makeData,
  RecordBatch,
  RecordBatchReader,
  Schema,
  Struct,
  Table,
  tableFromIPC,
  tableToIPC,
  Uint8,
  vectorFromArray,
} from "apache-arrow";

// An opt-in GPU gate: start Chrome with a dedicated profile/debug port first.
const endpoint = process.env.TRD_RENDERER_CDP;
const root = resolve(import.meta.dir, "..", "..", "..");
const golden = resolve(root, "crates", "trd-core", "tests", "golden");

async function fixtures() {
  const stage = Buffer.from(await Bun.file(resolve(golden, "stage2.arrow")).arrayBuffer());
  const firstVersion = stage.indexOf("0.0.7");
  const lastVersion = stage.lastIndexOf("0.0.7");
  expect(firstVersion).toBeGreaterThan(0);
  expect(lastVersion).toBeGreaterThan(firstVersion);
  const oldParams = Buffer.from(stage);
  oldParams.write("0.0.6", firstVersion);
  const oldMesh = Buffer.from(stage);
  oldMesh.write("0.0.6", lastVersion);
  const source = tableFromIPC(
    new Uint8Array(await Bun.file(resolve(golden, "stage1.arrow")).arrayBuffer()),
  );
  const vec3 = new FixedSizeList(3, new Field("item", new Float32(), false));
  const cameraFields = [
    ...["eye", "target", "up"].map((name) => new Field(name, vec3, false)),
    new Field("fovy", new Float32(), false),
  ];
  const cameraSchema = new Schema(cameraFields, source.schema.metadata);
  const params = new Table(
    cameraSchema,
    new RecordBatch(
      cameraSchema,
      makeData({
        type: new Struct(cameraFields),
        length: 3,
        children: [
          ...[
            [3, 2, 4],
            [0, 0, 0],
            [0, 1, 0],
          ].map((vector) => vectorFromArray([vector, vector, vector], vec3).data[0]),
          vectorFromArray(new Float32Array([0.7, 0.7, 0.7])).data[0],
        ],
      }),
    ),
  );
  const paramsBytes = tableToIPC(params, "stream");
  const fields = [...params.schema.fields, new Field("tonemap", new Uint8(), false)];
  const schema = new Schema(fields, params.schema.metadata);
  const toned = new Table(
    schema,
    params.batches.map(
      (batch) =>
        new RecordBatch(
          schema,
          makeData({
            type: new Struct(fields),
            length: batch.numRows,
            children: [
              ...batch.data.children,
              vectorFromArray(new Uint8Array(batch.numRows).fill(1)).data[0],
            ],
          }),
        ),
    ),
  );
  const streams = [];
  for (const reader of RecordBatchReader.readAll(stage)) {
    streams.push(new Table(reader.readAll()));
  }
  expect(streams.length).toBe(2);
  expect(streams[1].numRows).toBe(2);
  const mesh = streams[1].getChild("glb").get(0);
  const goldenHelper = await Bun.file(resolve(golden, "..", "support", "golden_image.rs")).text();
  const epsilon = goldenHelper.match(/const CHANNEL_EPS: u8 = (\d+);/);
  const fraction = goldenHelper.match(/const MAX_DIFF_FRACTION: f64 = ([\d.]+);/);
  expect(epsilon).not.toBeNull();
  expect(fraction).not.toBeNull();
  const goldenLimits = { channelEps: Number(epsilon[1]), maxDiffFraction: Number(fraction[1]) };
  const routes = new Map([
    ["/params.arrow", paramsBytes],
    ["/tonemap.arrow", tableToIPC(toned, "stream")],
    ["/old-params.arrow", oldParams],
    ["/old-mesh.arrow", oldMesh],
    ["/mesh.glb", mesh],
  ]);
  return { routes, goldenLimits };
}

test("renderer fixtures decode through the actual WASM document API without a GPU", async () => {
  const { routes } = await fixtures();
  const pkg = resolve(root, "crates", "trd-wasm", "pkg");
  const { default: init, ArrowSceneDocument } = await import(
    pathToFileURL(resolve(pkg, "trd_wasm.js")).href
  );
  await init({
    module_or_path: new Uint8Array(await Bun.file(resolve(pkg, "trd_wasm_bg.wasm")).arrayBuffer()),
  });
  for (const path of ["/params.arrow", "/tonemap.arrow"]) {
    const document = ArrowSceneDocument.fromArrow(routes.get(path));
    try {
      expect(document.frameCount()).toBe(3);
      expect(document.objectCount(0)).toBe(1);
      expect(document.meshIds()).toHaveLength(0);
      const camera = tableFromIPC(document.exportArrow());
      expect(camera.getChild("fovy").get(0)).toBeCloseTo(0.7);
      expect(Array.from(camera.getChild("eye").get(0))).toEqual([3, 2, 4]);
    } finally {
      document.free();
    }
  }
  const single = ArrowSceneDocument.fromArrowWithGlb(
    routes.get("/params.arrow"),
    routes.get("/mesh.glb"),
  );
  try {
    expect(single.meshIds()).toHaveLength(1);
    expect(single.objectCount(0)).toBe(1);
  } finally {
    single.free();
  }
  for (const path of ["/old-params.arrow", "/old-mesh.arrow"]) {
    expect(() => ArrowSceneDocument.fromArrow(routes.get(path))).toThrow("0.0.6");
  }
});

describe.skipIf(!endpoint)("current browser renderer contract (real WebGPU)", () => {
  test("canvas/offscreen pixels, lifecycle, model roundtrip and image IPC", async () => {
    const { routes, goldenLimits } = await fixtures();
    const server = Bun.serve({
      hostname: "127.0.0.1",
      port: 0,
      fetch(request) {
        const path = new URL(request.url).pathname;
        if (path === "/")
          return new Response("<!doctype html><title>trd renderer regression</title>", {
            headers: { "Content-Type": "text/html" },
          });
        if (routes.has(path)) return new Response(routes.get(path));
        let file;
        if (/^\/pkg\/trd_wasm(?:_bg\.wasm|\.js)$/.test(path)) {
          file = resolve(root, "crates", "trd-wasm", "pkg", path.slice(5));
        } else if (path === "/cases.js") {
          file = resolve(import.meta.dir, "renderer-cases.js");
        } else if (path === "/env.hdr") {
          file = resolve(root, "assets", "envmap", "uffizi-large.hdr");
        } else if (/^\/(?:frames\/)?[\w.-]+\.(?:arrow|png)$/.test(path)) {
          file = resolve(golden, path.slice(1));
        } else return new Response("not found", { status: 404 });
        return new Response(Bun.file(file));
      },
    });
    let socket;
    const pending = new Map();
    try {
      const pages = await (await fetch(`${endpoint}/json/list`)).json();
      const page = pages.find((entry) => entry.type === "page");
      expect(page).toBeDefined();
      socket = new WebSocket(page.webSocketDebuggerUrl);
      await new Promise((resolve, reject) => {
        socket.addEventListener("open", resolve, { once: true });
        socket.addEventListener("error", reject, { once: true });
      });
      let id = 0;
      socket.addEventListener("message", (event) => {
        const message = JSON.parse(event.data);
        const request = pending.get(message.id);
        if (!request) return;
        pending.delete(message.id);
        clearTimeout(request.timer);
        if (message.error) request.reject(new Error(JSON.stringify(message.error)));
        else request.resolve(message.result);
      });
      function send(method, params = {}) {
        return new Promise((resolve, reject) => {
          const seq = ++id;
          const timer = setTimeout(() => {
            pending.delete(seq);
            reject(new Error(`${method} timed out`));
          }, 60_000);
          pending.set(seq, { resolve, reject, timer });
          socket.send(JSON.stringify({ id: seq, method, params }));
        });
      }
      await send("Page.navigate", { url: String(server.url) });
      let loaded = false;
      for (let attempt = 0; attempt < 100; attempt++) {
        const state = await send("Runtime.evaluate", {
          expression: "location.origin",
          returnByValue: true,
        });
        if (state.result.value === server.url.origin) {
          loaded = true;
          break;
        }
        await Bun.sleep(50);
      }
      expect(loaded).toBe(true);
      const response = await send("Runtime.evaluate", {
        expression: `import("/cases.js").then(module => module.run(${JSON.stringify(goldenLimits)}))`,
        awaitPromise: true,
        returnByValue: true,
      });
      if (response.exceptionDetails) {
        const failure = await send("Runtime.evaluate", {
          expression: "globalThis.rendererFailure",
          returnByValue: true,
        });
        if (process.env.TRD_RENDERER_RESULTS && failure.result.value) {
          await Bun.write(
            `${process.env.TRD_RENDERER_RESULTS}.failure.json`,
            JSON.stringify(failure.result.value),
          );
        }
        throw new Error(JSON.stringify(response.exceptionDetails));
      }
      const report = response.result.value;
      const ipc = Buffer.from(report.ipc, "base64");
      const images = tableFromIPC(ipc);
      expect(images.schema.metadata.get("trd.protocol.version")).toBe("0.0.7");
      expect(images.numRows).toBe(3);
      for (let row = 0; row < images.numRows; row++) {
        const channels = ["r", "g", "b", "a"].map((name) =>
          images.getChild(name).get(row).toArray(),
        );
        const rgba = Buffer.alloc(320 * 180 * 4);
        for (let pixel = 0; pixel < 320 * 180; pixel++) {
          for (let channel = 0; channel < 4; channel++) {
            rgba[pixel * 4 + channel] = channels[channel][pixel];
          }
        }
        expect(createHash("sha256").update(rgba).digest("hex")).toBe(report.ipcHashes[row]);
      }
      delete report.ipc;
      report.imageIpcRows = images.numRows;
      report.timestamp = new Date().toISOString();
      if (process.env.TRD_RENDERER_RESULTS) {
        await Bun.write(process.env.TRD_RENDERER_RESULTS, JSON.stringify(report, null, 2));
      }
      console.log(JSON.stringify(report));
      await send("Page.navigate", { url: "about:blank" });
    } finally {
      socket?.close();
      for (const request of pending.values()) {
        clearTimeout(request.timer);
        request.reject(new Error("renderer test closed"));
      }
      server.stop(true);
    }
  }, 120_000);
});
