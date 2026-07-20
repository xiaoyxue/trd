// Textured-rendering demo (#20) for the wasm `CanvasRenderer` path: a full quad
// (protocol-0.0.3 mesh table with a per-vertex `uv` column) sampled from a small
// protocol-0.0.4 **texture table** (a 2x2 RGBA checker), driven by a params
// stream — the exact `[mesh][texture][params]` framing `scripts/obj_to_arrow.py`
// + `scripts/texture_to_arrow.py` + `scripts/jsonl_to_arrow.py` emit natively.
// The quad slowly spins in-plane so the four texels (red/green/blue/white) stay
// visible, proving the browser renderer samples the bound texture at each UV.
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
  Uint8,
  Uint32,
  vectorFromArray,
} from "apache-arrow";
import init, { CanvasRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };

const canvas = document.getElementById("trd-canvas");
const status = document.getElementById("trd-status");
if (!(canvas instanceof HTMLCanvasElement)) {
  throw new Error("missing #trd-canvas element");
}
if (!(status instanceof HTMLElement)) {
  throw new Error("missing #trd-status element");
}
const canvasElement = canvas;
const statusElement = status;

// --- Mesh table (0.0.3): a quad with `position` + `uv` (+ `index`). ----------
const f32x3 = new FixedSizeList(3, new Field("item", new Float32(), false));
const f32x2 = new FixedSizeList(2, new Field("item", new Float32(), false));
const positionType = new List(new Field("item", f32x3, false));
const uvType = new List(new Field("item", f32x2, false));
const indexType = new List(new Field("item", new Uint32(), false));
const meshSchema = new Schema(
  [
    new Field("position", positionType, false),
    new Field("uv", uvType, false),
    new Field("index", indexType, false),
  ],
  new Map([["trd.protocol.version", "0.0.3"]]),
);

type Vec3 = readonly [number, number, number];
type Vec2 = readonly [number, number];

// A unit quad in the z=0 plane. UVs use the top-left texel origin (v grows down,
// matching `scripts/texture_to_arrow.py`'s `[u, 1 - v]`), so the quad's top-left
// corner samples texel (row 0, col 0).
const QUAD_POSITIONS: readonly Vec3[] = [
  [-1, -1, 0],
  [1, -1, 0],
  [1, 1, 0],
  [-1, 1, 0],
];
const QUAD_UVS: readonly Vec2[] = [
  [0, 1],
  [1, 1],
  [1, 0],
  [0, 0],
];
const QUAD_INDICES: readonly number[] = [0, 1, 2, 0, 2, 3];

function meshStreamBytes(): Uint8Array {
  // One row = one mesh: each column's single row is the whole vertex/index list.
  const position = vectorFromArray([QUAD_POSITIONS], positionType).data[0];
  const uv = vectorFromArray([QUAD_UVS], uvType).data[0];
  const index = vectorFromArray([QUAD_INDICES], indexType).data[0];
  if (!position || !uv || !index) {
    throw new Error("mesh Arrow vector construction produced no data");
  }
  const batch = new RecordBatch(
    meshSchema,
    makeData({
      type: new Struct(meshSchema.fields),
      length: 1,
      nullCount: 0,
      children: [position, uv, index],
    }),
  );
  return RecordBatchStreamWriter.writeAll([batch]).toUint8Array(true);
}

// --- Texture table (0.0.4): a 2x2 RGBA checker as a fixed_shape_tensor. -------
const TEX_W = 2;
const TEX_H = 2;
// Row-major, top-left origin: red, green / blue, white.
const CHECKER_RGBA: readonly number[] = [
  255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
];
const rgbaType = new FixedSizeList(TEX_W * TEX_H * 4, new Field("item", new Uint8(), false));
// The `rgba` field carries the canonical fixed_shape_tensor extension so its
// [H, W, 4] shape is self-describing (trd-core reads H/W from this metadata).
const rgbaField = new Field(
  "rgba",
  rgbaType,
  false,
  new Map([
    ["ARROW:extension:name", "arrow.fixed_shape_tensor"],
    ["ARROW:extension:metadata", JSON.stringify({ shape: [TEX_H, TEX_W, 4] })],
  ]),
);
const textureSchema = new Schema([rgbaField], new Map([["trd.protocol.version", "0.0.4"]]));

function textureStreamBytes(): Uint8Array {
  const rgba = vectorFromArray([CHECKER_RGBA], rgbaType).data[0];
  if (!rgba) {
    throw new Error("texture Arrow vector construction produced no data");
  }
  const batch = new RecordBatch(
    textureSchema,
    makeData({
      type: new Struct(textureSchema.fields),
      length: 1,
      nullCount: 0,
      children: [rgba],
    }),
  );
  return RecordBatchStreamWriter.writeAll([batch]).toUint8Array(true);
}

// --- Params table (0.0.3): per-frame `model` + a static CG camera. -----------
const mat4 = new FixedSizeList(16, new Field("item", new Float32(), false));
const paramsSchema = new Schema(
  [
    new Field("center", f32x2, false),
    new Field("size", f32x2, false),
    new Field("theta", new Float32(), false),
    new Field("model", mat4, false),
    new Field("eye", f32x3, false),
    new Field("target", f32x3, false),
    new Field("up", f32x3, false),
    new Field("fovy", new Float32(), false),
    new Field("aspect", new Float32(), false),
  ],
  new Map([["trd.protocol.version", "0.0.3"]]),
);

const ZERO2: Vec2 = [0, 0];
const ONE2: Vec2 = [1, 1];
// Head-on camera; the quad (centered + fit to 2.0 by the preview transform) fills
// the view at eye distance 3 with fovy ~0.8.
const EYE: Vec3 = [0, 0, 3];
const TARGET: Vec3 = [0, 0, 0];
const UP: Vec3 = [0, 1, 0];
const FOVY = 0.8;

// Column-major `rotate_z(theta)` — an in-plane spin that keeps the quad (and all
// four texels) facing the camera at every angle.
function rotateZ(theta: number): number[] {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function frameBatch(theta: number): RecordBatch {
  const center = vectorFromArray([ZERO2], f32x2).data[0];
  const size = vectorFromArray([ONE2], f32x2).data[0];
  const thetaCol = vectorFromArray([0], new Float32()).data[0];
  const model = vectorFromArray([rotateZ(theta)], mat4).data[0];
  const eye = vectorFromArray([EYE], f32x3).data[0];
  const target = vectorFromArray([TARGET], f32x3).data[0];
  const up = vectorFromArray([UP], f32x3).data[0];
  const fovy = vectorFromArray([FOVY], new Float32()).data[0];
  const aspect = vectorFromArray([1], new Float32()).data[0];
  if (!center || !size || !thetaCol || !model || !eye || !target || !up || !fovy || !aspect) {
    throw new Error("params Arrow vector construction produced no data");
  }
  return new RecordBatch(
    paramsSchema,
    makeData({
      type: new Struct(paramsSchema.fields),
      length: 1,
      nullCount: 0,
      children: [center, size, thetaCol, model, eye, target, up, fovy, aspect],
    }),
  );
}

async function run(): Promise<void> {
  await init({ module_or_path: wasmUrl });

  const renderer = await CanvasRenderer.create(canvasElement);
  // Sample the bound texture at each vertex UV (vs. per-vertex color).
  renderer.setTextured(true);

  // Deliver the leading [mesh][texture] tables before any params frame so the
  // renderer (built lazily on the first frame) binds the checker as its albedo.
  renderer.pushIpc(meshStreamBytes());
  renderer.pushIpc(textureStreamBytes());

  // Stream params frames one at a time, paced by requestAnimationFrame, spinning
  // the quad in-plane. Each pushIpc chunk continues the single params sub-stream
  // (a params stream must stay a single Arrow IPC stream, so it is chunked rather
  // than re-opened per frame). The pump runs for the life of the page.
  const writer = new RecordBatchStreamWriter({ compressionType: null });
  void (async () => {
    for await (const chunk of writer) {
      renderer.pushIpc(chunk);
    }
  })().catch((error: unknown) => {
    statusElement.textContent = `textured quad: error — ${String(error)}`;
    throw error;
  });

  statusElement.textContent = "textured quad: streaming";
  let frame = 0;
  const step = () => {
    writer.write(frameBatch((frame * Math.PI) / 90));
    frame += 1;
    statusElement.dataset.texturedFrames = String(frame);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

await run();

export {};
