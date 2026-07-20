import {
  Field,
  FixedSizeList,
  Float32,
  List,
  makeData,
  RecordBatch,
  RecordBatchStreamWriter,
  Schema,
  Struct,
  Uint32,
  vectorFromArray,
} from "apache-arrow";
import init, { CanvasRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };
// The Stanford bunny OBJ (colorless), parsed in-browser into a protocol-0.0.3
// **mesh table** — exactly as `scripts/obj_to_arrow.py` encodes it natively.
import bunnyUrl from "../../assets/meshes/bunny.obj" with { type: "file" };
// The 45° bird's-eye *dolly* camera capstone params (#49): each row carries a
// `rotate_y(theta_i)` model **plus** a CG camera (eye/target/up/fovy/aspect),
// authored by `examples/bunny_dolly.py`. The browser thus renders the identical
// scene the native `render.sh --wireframe --aabb --axes --mesh
// assets/meshes/bunny.obj examples/frames.bunny_dolly.cg.jsonl` does.
import framesUrl from "../../examples/frames.bunny_dolly.cg.jsonl" with { type: "file" };

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
const vec3 = new FixedSizeList(3, new Field("item", new Float32(), false));
// Column-major 4x4 Mat4 = the mesh's per-frame model transform (protocol 0.0.3).
const mat4 = new FixedSizeList(16, new Field("item", new Float32(), false));
// The params stream that follows the leading mesh table (0.0.3): the legacy
// center/size/theta + per-frame `model`, plus the CG dolly-camera columns
// (eye/target/up/fovy/aspect) — the exact schema `scripts/jsonl_to_arrow.py`
// emits for `examples/frames.bunny_dolly.cg.jsonl`.
const schema = new Schema(
  [
    new Field("center", vec2, false),
    new Field("size", vec2, false),
    new Field("theta", new Float32(), false),
    new Field("model", mat4, false),
    new Field("eye", vec3, false),
    new Field("target", vec3, false),
    new Field("up", vec3, false),
    new Field("fovy", new Float32(), false),
    new Field("aspect", new Float32(), false),
  ],
  new Map([["trd.protocol.version", "0.0.3"]]),
);

const f32Item = new Field("item", new Float32(), false);
const geometryType = new List(new Field("item", new FixedSizeList(3, f32Item), false));
const indexType = new List(new Field("item", new Uint32(), false));
// The bunny carries no vertex colors, so the mesh table is `position` + `index`
// only (obj_to_arrow.py omits the optional `color` column for a colorless OBJ;
// trd then defaults every vertex to DEFAULT_COLOR).
const meshSchema = new Schema(
  [new Field("position", geometryType, false), new Field("index", indexType, false)],
  new Map([["trd.protocol.version", "0.0.3"]]),
);

type Vec3 = readonly [number, number, number];

/// Parses OBJ text into position triples + a triangle index list, mirroring
/// `scripts/obj_to_arrow.py`: only `v x y z` (positions) and `f` (faces) are
/// read; each face-vertex reference `a/b/c` uses the position index (`a`) only,
/// 1-based (negative = relative to the end), and polygons are fan-triangulated.
function parseObj(text: string): { positions: Vec3[]; indices: number[] } {
  const positions: Vec3[] = [];
  const indices: number[] = [];
  for (const line of text.split("\n")) {
    if (line.startsWith("v ")) {
      const coords = line.slice(2).trim().split(/\s+/);
      positions.push([Number(coords[0]), Number(coords[1]), Number(coords[2])]);
    } else if (line.startsWith("f ")) {
      const refs = line
        .slice(2)
        .trim()
        .split(/\s+/)
        .map((token) => {
          const raw = Number.parseInt(token.split("/")[0] ?? "", 10);
          return raw > 0 ? raw - 1 : positions.length + raw;
        });
      for (let i = 1; i + 1 < refs.length; i += 1) {
        const a = refs[0];
        const b = refs[i];
        const c = refs[i + 1];
        if (a === undefined || b === undefined || c === undefined) {
          throw new Error(`invalid OBJ face: ${line}`);
        }
        indices.push(a, b, c);
      }
    }
  }
  return { positions, indices };
}

/// Serializes the parsed bunny as a one-row `[mesh]` Arrow IPC stream (schema +
/// batch + end-of-stream). Pushed to the renderer before any params frame, so the
/// `InputSession` decodes it as the leading mesh table and the `MeshRenderer`
/// renders it (centered + scaled to fit) driven by the per-frame model.
function meshStreamBytes(positions: readonly Vec3[], indices: readonly number[]): Uint8Array {
  // One row = one mesh: each column's single row is the whole vertex/index list
  // (hence the extra array nesting around the geometry).
  const position = vectorFromArray([positions], geometryType).data[0];
  const index = vectorFromArray([indices], indexType).data[0];
  if (!position || !index) {
    throw new Error("mesh Arrow vector construction produced no data");
  }
  const batch = new RecordBatch(
    meshSchema,
    makeData({
      type: new Struct(meshSchema.fields),
      length: 1,
      nullCount: 0,
      children: [position, index],
    }),
  );
  return RecordBatchStreamWriter.writeAll([batch]).toUint8Array(true);
}

// A frame carries the mesh's 4x4 model matrix (16 column-major floats) and the
// per-frame CG dolly camera; the legacy center/size/theta columns are still
// required on the wire and filled with the identity below.
type Frame = Readonly<{
  model: readonly number[];
  eye: Vec3;
  target: Vec3;
  up: Vec3;
  fovy: number;
  aspect: number;
}>;

const ZERO2: readonly [number, number] = [0, 0];
const ONE2: readonly [number, number] = [1, 1];

// A default CG camera for the two-row smoke batch (which authors only models).
const SMOKE_CAMERA = {
  eye: [1.2, 0.9, 2.6] as Vec3,
  target: [0, 0, 0] as Vec3,
  up: [0, 1, 0] as Vec3,
  fovy: Math.PI / 4,
  aspect: 1,
} as const;

/// Column-major `translate(center) . rotate_z(theta) . scale(size)` — the same
/// 4x4 model matrix trd-core synthesizes (glam) and the producer emits. Used to
/// author the two-frame smoke batch below directly as matrices.
function modelMatrix(
  center: readonly [number, number],
  size: readonly [number, number],
  theta: number,
): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const [sx, sy] = size;
  const [tx, ty] = center;
  return [sx * c, sx * s, 0, 0, -sy * s, sy * c, 0, 0, 0, 0, 1, 0, tx, ty, 0, 1];
}

function frameBatch(frames: readonly Frame[]): RecordBatch {
  const center = vectorFromArray(
    frames.map(() => ZERO2),
    vec2,
  ).data[0];
  const size = vectorFromArray(
    frames.map(() => ONE2),
    vec2,
  ).data[0];
  const theta = vectorFromArray(
    frames.map(() => 0),
    new Float32(),
  ).data[0];
  const model = vectorFromArray(
    frames.map((frame) => frame.model),
    mat4,
  ).data[0];
  const eye = vectorFromArray(
    frames.map((frame) => frame.eye),
    vec3,
  ).data[0];
  const target = vectorFromArray(
    frames.map((frame) => frame.target),
    vec3,
  ).data[0];
  const up = vectorFromArray(
    frames.map((frame) => frame.up),
    vec3,
  ).data[0];
  const fovy = vectorFromArray(
    frames.map((frame) => frame.fovy),
    new Float32(),
  ).data[0];
  const aspect = vectorFromArray(
    frames.map((frame) => frame.aspect),
    new Float32(),
  ).data[0];

  if (!center || !size || !theta || !model || !eye || !target || !up || !fovy || !aspect) {
    throw new Error("Arrow vector construction produced no data");
  }

  return new RecordBatch(
    schema,
    makeData({
      type: new Struct(schema.fields),
      length: frames.length,
      nullCount: 0,
      children: [center, size, theta, model, eye, target, up, fovy, aspect],
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

/// Reads a 3-float array field from a parsed JSONL row.
function readVec3(row: Record<string, unknown>, key: string): Vec3 {
  const value = row[key];
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`invalid frame row (expected 3-float ${key})`);
  }
  return [Number(value[0]), Number(value[1]), Number(value[2])];
}

/// Loads `examples/frames.bunny_dolly.cg.jsonl` — the 45° bird's-eye *dolly*
/// camera capstone (#49): each row's `rotate_y(theta_i)` model + CG camera
/// (eye/target/up/fovy/aspect), so the browser renders the identical animation
/// the native CLI does from the same source of truth.
async function loadFrames(): Promise<Frame[]> {
  const response = await fetch(framesUrl);
  if (!response.ok) {
    throw new Error(`failed to load frames.bunny_dolly.cg.jsonl: ${response.status}`);
  }
  const text = await response.text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => {
      const parsed: unknown = JSON.parse(line);
      if (typeof parsed !== "object" || parsed === null) {
        throw new Error(`invalid frame row: ${line}`);
      }
      const row = parsed as Record<string, unknown>;
      const model = row.model;
      if (!Array.isArray(model) || model.length !== 16) {
        throw new Error(`invalid frame row (expected 16-float model): ${line}`);
      }
      const fovy = Number(row.fovy);
      const aspect = Number(row.aspect);
      if (!Number.isFinite(fovy) || !Number.isFinite(aspect)) {
        throw new Error(`invalid frame row (expected fovy/aspect scalars): ${line}`);
      }
      return {
        model: model.map(Number),
        eye: readVec3(row, "eye"),
        target: readVec3(row, "target"),
        up: readVec3(row, "up"),
        fovy,
        aspect,
      };
    });
}

async function run(): Promise<void> {
  await init({ module_or_path: wasmUrl });

  // The render resolution = the canvas drawing-buffer size (default 1024x1024,
  // matching the dolly camera's `aspect = 1.0`). Override it with `?size=N` for a
  // square NxN buffer; the CanvasRenderer reads width/height at creation time, so
  // this must run before `create`.
  const query = new URLSearchParams(window.location.search);
  const sizeParam = Number(query.get("size"));
  if (Number.isFinite(sizeParam) && sizeParam >= 16 && sizeParam <= 4096) {
    const size = Math.floor(sizeParam);
    canvasElement.width = size;
    canvasElement.height = size;
  }

  const renderer = await CanvasRenderer.create(canvasElement);

  // Match the native `--wireframe --aabb --axes` config: draw the bunny as a
  // wireframe and overlay its AABB box + the world coordinate-axes gizmo (the
  // same DrawableObject scene those flags build). Fetch + parse the bunny OBJ and
  // deliver it as the leading mesh table before any params frame.
  renderer.setWireframe(true);
  renderer.setShowAabb(true);
  renderer.setShowAxes(true);

  const bunnyResponse = await fetch(bunnyUrl);
  if (!bunnyResponse.ok) {
    throw new Error(`failed to load bunny.obj: ${bunnyResponse.status}`);
  }
  const { positions, indices } = parseObj(await bunnyResponse.text());
  if (positions.length === 0 || indices.length === 0) {
    throw new Error("bunny OBJ parsed to an empty mesh");
  }
  renderer.pushIpc(meshStreamBytes(positions, indices));

  const writer = new RecordBatchStreamWriter({ compressionType: null });
  const smoke = query.get("smoke") === "1";
  const rate = Number(query.get("benchmarkRate"));
  const benchmark = rate === 60 || rate === 120;
  // Playback frame rate (default 24, matching the native GIF); `?fps=N` overrides.
  // The benchmark path keeps its own fixed 60/120 pacing.
  const fpsParam = Number(query.get("fps"));
  const fps = Number.isFinite(fpsParam) && fpsParam >= 1 && fpsParam <= 240 ? fpsParam : 24;
  const frameRate = benchmark ? rate : fps;
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
    { model: modelMatrix([-0.35, 0], [0.45, 0.45], 0), ...SMOKE_CAMERA },
    { model: modelMatrix([0.35, 0], [0.45, 0.45], Math.PI / 2), ...SMOKE_CAMERA },
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
  // present, paced to `frameRate` (default 24 fps, `?fps=N` override) via
  // setTimeout so playback matches the native GIF's timing.
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

    nextDeadline += 1000 / frameRate;
    const delay = Math.max(0, nextDeadline - performance.now());
    window.setTimeout(() => {
      void schedule().catch(reportError);
    }, delay);
  }

  nextDeadline += 1000 / frameRate;
  window.setTimeout(
    () => {
      void schedule().catch(reportError);
    },
    Math.max(0, nextDeadline - performance.now()),
  );
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
