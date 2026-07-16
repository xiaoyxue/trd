import {
  Field,
  FixedSizeList,
  Float32,
  makeData,
  RecordBatch,
  RecordBatchStreamWriter,
  Schema,
  Struct,
  vectorFromArray,
} from "apache-arrow";
import init, { CanvasRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };
// The same frame data the native CLI/window consume, so all three front-ends
// render the identical animation from one shared source of truth.
import framesUrl from "../../examples/frames.jsonl" with { type: "file" };

const canvas = document.getElementById("trd-canvas");
const status = document.getElementById("trd-status");

if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error('expected a <canvas id="trd-canvas"> element');
}
if (!(status instanceof HTMLParagraphElement)) {
  throw new Error('expected a <p id="trd-status"> element');
}

const canvasElement = canvas;
const statusElement = status;

const vec2 = new FixedSizeList(2, new Field("item", new Float32(), false));
const schema = new Schema(
  [
    new Field("center", vec2, false),
    new Field("size", vec2, false),
    new Field("theta", new Float32(), false),
  ],
  new Map([["trd.protocol.version", "0.0.1"]]),
);

type Frame = Readonly<{
  center: readonly [number, number];
  size: readonly [number, number];
  theta: number;
}>;

function frameBatch(frames: readonly Frame[]): RecordBatch {
  const center = vectorFromArray(
    frames.map((frame) => frame.center),
    vec2,
  );
  const size = vectorFromArray(
    frames.map((frame) => frame.size),
    vec2,
  );
  const theta = vectorFromArray(
    frames.map((frame) => frame.theta),
    new Float32(),
  );
  const centerData = center.data[0];
  const sizeData = size.data[0];
  const thetaData = theta.data[0];

  if (!centerData || !sizeData || !thetaData) {
    throw new Error("Arrow vector construction produced no data");
  }

  return new RecordBatch(
    schema,
    makeData({
      type: new Struct(schema.fields),
      length: frames.length,
      nullCount: 0,
      children: [centerData, sizeData, thetaData],
    }),
  );
}

function percentile(values: readonly number[], quantile: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * quantile))] ?? 0;
}

function summary(name: string, values: readonly number[]) {
  return {
    name,
    count: values.length,
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
  };
}

function addMeasure(name: string, duration: number): void {
  performance.measure(name, { start: performance.now(), duration });
}

/// Loads the shared `examples/frames.jsonl` frame sequence — the same input data
/// the native CLI/window render — so the browser plays the identical animation.
async function loadFrames(): Promise<Frame[]> {
  const response = await fetch(framesUrl);
  if (!response.ok) {
    throw new Error(`failed to load frames.jsonl: ${response.status}`);
  }
  const text = await response.text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const row: unknown = JSON.parse(line);
      if (
        typeof row !== "object" ||
        row === null ||
        !("center" in row) ||
        !("size" in row) ||
        !("theta" in row)
      ) {
        throw new Error(`invalid frame row: ${line}`);
      }
      const frame = row as Frame;
      return { center: frame.center, size: frame.size, theta: frame.theta };
    });
}

async function run(): Promise<void> {
  await init({ module_or_path: wasmUrl });
  const renderer = await CanvasRenderer.create(canvasElement);
  const writer = new RecordBatchStreamWriter({ compressionType: null });
  const query = new URLSearchParams(window.location.search);
  const smoke = query.get("smoke") === "1";
  const rate = Number(query.get("benchmarkRate"));
  const benchmark = rate === 60 || rate === 120;
  const totalFrames = benchmark ? 600 : 300;

  const generation = [] as number[];
  const pushTotals = [] as number[];
  const renderSubmit = [] as number[];
  const transferPlusDecode = [] as number[];
  let acknowledge: ((rows: number) => void) | undefined;
  let rejectAcknowledge: ((reason: unknown) => void) | undefined;
  let pendingPush = 0;
  let pendingRender = 0;

  const pump = (async () => {
    for await (const chunk of writer) {
      const renderStart = performance.getEntriesByName("trd.canvas.render-submit").length;
      const start = performance.now();
      const rows = renderer.pushIpc(chunk);
      pendingPush += performance.now() - start;

      if (rows > 0) {
        pendingRender += performance
          .getEntriesByName("trd.canvas.render-submit")
          .slice(renderStart)
          .reduce((sum, entry) => sum + entry.duration, 0);
        performance.clearMeasures();

        const derived = Math.max(0, pendingPush - pendingRender);
        pushTotals.push(pendingPush);
        renderSubmit.push(pendingRender);
        transferPlusDecode.push(derived);
        addMeasure("trd.pushIpc.total", pendingPush);
        addMeasure("trd.transfer-plus-decode", derived);

        pendingPush = 0;
        pendingRender = 0;

        const resolve = acknowledge;
        acknowledge = undefined;
        rejectAcknowledge = undefined;
        resolve?.(rows);
      }
    }
  })().catch((error: unknown) => {
    rejectAcknowledge?.(error);
    acknowledge = undefined;
    rejectAcknowledge = undefined;
    throw error;
  });

  function append(batch: RecordBatch, expectedRows: number): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      acknowledge = (rows) => {
        if (rows === expectedRows) {
          resolve();
        } else {
          reject(new Error(`expected ${expectedRows} rendered rows, received ${rows}`));
        }
      };
      rejectAcknowledge = reject;
      writer.write(batch);
    });
  }

  async function appendOne(frame: Frame): Promise<void> {
    const start = performance.now();
    const batch = frameBatch([frame]);
    const duration = performance.now() - start;
    generation.push(duration);
    addMeasure("trd.arrow-js.generation", duration);
    await append(batch, 1);
  }

  const smokeStart = performance.now();
  const smokeBatch = frameBatch([
    { center: [-0.35, 0], size: [0.45, 0.45], theta: 0 },
    { center: [0.35, 0], size: [0.45, 0.45], theta: Math.PI / 2 },
  ]);
  const smokeDuration = performance.now() - smokeStart;
  generation.push(smokeDuration);
  addMeasure("trd.arrow-js.generation", smokeDuration);
  await append(smokeBatch, 2);
  statusElement.dataset.rowsRendered = "2";
  statusElement.textContent = "smoke rows rendered: 2";

  if (smoke) {
    writer.finish();
    await pump;
    renderer.finish();
    statusElement.dataset.state = "finished";
    return;
  }

  // The shared frame sequence (same as native); cycle through it, one frame per
  // present. Speed = frame_rate carried in the data via the number of frames;
  // pacing is the browser's requestAnimationFrame (its refresh rate) — no fps
  // knob, matching the native "the stream is the animation" model.
  const frames = await loadFrames();
  if (frames.length === 0) {
    throw new Error("shared frame stream is empty");
  }

  const started = performance.now();
  let nextDeadline = started;
  let completed = 0;

  async function schedule(): Promise<void> {
    await appendOne(frames[completed % frames.length] as Frame);
    completed += 1;

    if (completed === totalFrames) {
      writer.finish();
      await pump;
      renderer.finish();
      const elapsedSeconds = (performance.now() - started) / 1000;
      statusElement.dataset.state = "finished";
      statusElement.textContent = "finished";
      console.table([
        summary("Arrow generation", generation),
        summary("pushIpc total", pushTotals),
        summary("render-submit", renderSubmit),
        summary("transfer plus decode", transferPlusDecode),
        { name: "achieved batches/sec", count: completed / elapsedSeconds, p50: 0, p95: 0, p99: 0 },
      ]);
      return;
    }

    if (benchmark) {
      nextDeadline += 1000 / rate;
      const delay = Math.max(0, nextDeadline - performance.now());
      window.setTimeout(() => {
        void schedule().catch(reportError);
      }, delay);
    } else {
      requestAnimationFrame(() => {
        void schedule().catch(reportError);
      });
    }
  }

  if (benchmark) {
    nextDeadline += 1000 / rate;
    window.setTimeout(
      () => {
        void schedule().catch(reportError);
      },
      Math.max(0, nextDeadline - performance.now()),
    );
  } else {
    requestAnimationFrame(() => {
      void schedule().catch(reportError);
    });
  }
}

function reportError(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  const status = document.getElementById("trd-status");
  if (status instanceof HTMLParagraphElement) {
    status.dataset.state = "error";
    status.textContent = `error: ${message}`;
  }
  console.error("trd canvas demo failed:", error);
}

void run().catch(reportError);
