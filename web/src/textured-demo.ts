// Textured-rendering demo (#20) for the wasm `CanvasRenderer` path — the browser
// twin of the native `render.sh --cli --mesh bunny.obj --texture bunny_uv_map1.jpg
// --aabb --axes …` render. It parses the UV-mapped Stanford bunny OBJ in-browser
// into a protocol-0.0.3 **mesh table** (`position` + `uv` + `index`, splitting
// vertices at UV seams exactly as `scripts/obj_to_arrow.py` does), decodes the
// JPEG albedo atlas into a protocol-0.0.4 **texture table** (an `arrow.
// fixed_shape_tensor<uint8>` of shape `[H, W, 4]`, matching `scripts/
// texture_to_arrow.py`), and drives them with the 45° bird's-eye *dolly* camera
// params (#49). The mesh is drawn textured (sampling the atlas at each vertex UV)
// with its AABB box + the world coordinate-axes gizmo overlaid — the exact
// `[mesh][texture][params]` framing + `--texture --aabb --axes` scene the CLI
// produces natively, so the browser and headless renders match.
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
// The UV-mapped Stanford bunny OBJ + its albedo atlas (checker background, eyes +
// pink nose/ears) and the shared dolly-camera params — the same assets the native
// CLI textured render consumes.
import bunnyUrl from "../../assets/meshes/bunny_with_texture/bunny.obj" with { type: "file" };
import textureUrl from "../../assets/meshes/bunny_with_texture/bunny_uv_map1.jpg" with {
  type: "file",
};
import framesUrl from "../../examples/frames.bunny_dolly.cg.jsonl" with { type: "file" };

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

type Vec3 = readonly [number, number, number];
type Vec2 = readonly [number, number];

// GPUs cap texture dimensions, so downscale the 3072² atlas to match the CLI's
// `--max-size 2048` default before uploading.
const MAX_TEXTURE_SIZE = 2048;

// --- Mesh table (0.0.3): the bunny with `position` + `uv` (+ `index`). --------
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

// Resolves an OBJ face index token: 1-based, negative = relative to the current
// count; returns a 0-based index, or -1 when the token is absent (e.g. no `vt`).
function resolveIndex(raw: number, count: number): number {
  if (!Number.isFinite(raw)) {
    return -1;
  }
  return raw > 0 ? raw - 1 : count + raw;
}

// Parses OBJ text into split `position` + `uv` + `index` arrays, mirroring
// `scripts/obj_to_arrow.py`: OBJ indexes positions and texcoords independently
// (`f v/vt/vn`), but trd carries one uv per vertex, so each unique `(position,
// texcoord)` corner becomes its own output vertex (indices remapped). `vt` v runs
// bottom-up, so uv is emitted V-flipped as `[u, 1 - v]` to the top-left texel
// origin. Polygons are fan-triangulated.
function parseTexturedObj(text: string): {
  positions: Vec3[];
  uvs: Vec2[];
  indices: number[];
} {
  const rawPositions: Vec3[] = [];
  const rawTexcoords: Vec2[] = [];
  const positions: Vec3[] = [];
  const uvs: Vec2[] = [];
  const indices: number[] = [];
  const cornerMap = new Map<string, number>();

  function corner(positionIndex: number, texcoordIndex: number): number {
    const key = `${positionIndex}/${texcoordIndex}`;
    const existing = cornerMap.get(key);
    if (existing !== undefined) {
      return existing;
    }
    const position = rawPositions[positionIndex];
    if (position === undefined) {
      throw new Error(`OBJ face references missing vertex ${positionIndex}`);
    }
    const index = positions.length;
    cornerMap.set(key, index);
    positions.push(position);
    const texcoord = texcoordIndex >= 0 ? rawTexcoords[texcoordIndex] : undefined;
    uvs.push(texcoord ? [texcoord[0], 1 - texcoord[1]] : [0, 0]);
    return index;
  }

  for (const line of text.split("\n")) {
    if (line.startsWith("v ")) {
      const coords = line.slice(2).trim().split(/\s+/);
      rawPositions.push([Number(coords[0]), Number(coords[1]), Number(coords[2])]);
    } else if (line.startsWith("vt ")) {
      const coords = line.slice(3).trim().split(/\s+/);
      rawTexcoords.push([Number(coords[0]), Number(coords[1])]);
    } else if (line.startsWith("f ")) {
      const refs = line
        .slice(2)
        .trim()
        .split(/\s+/)
        .map((token) => {
          const fields = token.split("/");
          const positionIndex = resolveIndex(
            Number.parseInt(fields[0] ?? "", 10),
            rawPositions.length,
          );
          const texcoordIndex = resolveIndex(
            Number.parseInt(fields[1] ?? "", 10),
            rawTexcoords.length,
          );
          return { positionIndex, texcoordIndex };
        });
      for (let i = 1; i + 1 < refs.length; i += 1) {
        const a = refs[0];
        const b = refs[i];
        const c = refs[i + 1];
        if (a === undefined || b === undefined || c === undefined) {
          throw new Error(`invalid OBJ face: ${line}`);
        }
        indices.push(
          corner(a.positionIndex, a.texcoordIndex),
          corner(b.positionIndex, b.texcoordIndex),
          corner(c.positionIndex, c.texcoordIndex),
        );
      }
    }
  }
  return { positions, uvs, indices };
}

function meshStreamBytes(
  positions: readonly Vec3[],
  uvs: readonly Vec2[],
  indices: readonly number[],
): Uint8Array {
  // One row = one mesh: each column's single row is the whole vertex/index list.
  const position = vectorFromArray([positions], positionType).data[0];
  const uv = vectorFromArray([uvs], uvType).data[0];
  const index = vectorFromArray([indices], indexType).data[0];
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

// --- Texture table (0.0.4): the atlas as an `arrow.fixed_shape_tensor<uint8>`. -
// Decodes the JPEG to row-major, top-left-origin RGBA (matching PIL's default in
// `scripts/texture_to_arrow.py`), downscaling the longest side to
// `MAX_TEXTURE_SIZE` while preserving aspect ratio.
async function decodeTextureRgba(
  url: string,
): Promise<{ width: number; height: number; rgba: Uint8Array }> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to load texture ${url}: ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob());
  let width = bitmap.width;
  let height = bitmap.height;
  const longest = Math.max(width, height);
  if (longest > MAX_TEXTURE_SIZE) {
    const scale = MAX_TEXTURE_SIZE / longest;
    width = Math.max(1, Math.round(width * scale));
    height = Math.max(1, Math.round(height * scale));
  }
  const offscreen = new OffscreenCanvas(width, height);
  const context = offscreen.getContext("2d");
  if (!context) {
    throw new Error("failed to acquire OffscreenCanvas 2D context");
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const imageData = context.getImageData(0, 0, width, height);
  return { width, height, rgba: new Uint8Array(imageData.data.buffer.slice(0)) };
}

function textureStreamBytes(width: number, height: number, rgba: Uint8Array): Uint8Array {
  const listSize = width * height * 4;
  if (rgba.length !== listSize) {
    throw new Error(`texture RGBA length ${rgba.length} != ${listSize} (${width}x${height})`);
  }
  const rgbaType = new FixedSizeList(listSize, new Field("item", new Uint8(), false));
  // The `rgba` field carries the canonical fixed_shape_tensor extension so its
  // [H, W, 4] shape is self-describing (trd-core reads H/W from this metadata).
  const rgbaField = new Field(
    "rgba",
    rgbaType,
    false,
    new Map([
      ["ARROW:extension:name", "arrow.fixed_shape_tensor"],
      ["ARROW:extension:metadata", JSON.stringify({ shape: [height, width, 4] })],
    ]),
  );
  const textureSchema = new Schema([rgbaField], new Map([["trd.protocol.version", "0.0.4"]]));
  // Build the one-row FixedSizeList<uint8>[H*W*4] directly from the typed array
  // (a JS number[] round-trip would be prohibitive at multi-megapixel sizes).
  const rgbaData = makeData({
    type: rgbaType,
    length: 1,
    nullCount: 0,
    child: makeData({ type: new Uint8(), data: rgba }),
  });
  const batch = new RecordBatch(
    textureSchema,
    makeData({
      type: new Struct(textureSchema.fields),
      length: 1,
      nullCount: 0,
      children: [rgbaData],
    }),
  );
  return RecordBatchStreamWriter.writeAll([batch]).toUint8Array(true);
}

// --- Params table (0.0.3): per-frame `model` + the CG dolly camera. -----------
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

type Frame = Readonly<{
  model: readonly number[];
  eye: Vec3;
  target: Vec3;
  up: Vec3;
  fovy: number;
  aspect: number;
}>;

function readVec3(row: Record<string, unknown>, key: string): Vec3 {
  const value = row[key];
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`invalid frame row (expected 3-float ${key})`);
  }
  return [Number(value[0]), Number(value[1]), Number(value[2])];
}

// Loads `examples/frames.bunny_dolly.cg.jsonl` — the 45° bird's-eye dolly camera
// capstone (#49): each row's `rotate_y(theta_i)` model + CG camera, the same
// source of truth the native CLI render uses.
async function loadFrames(url: string): Promise<Frame[]> {
  const response = await fetch(url);
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

function frameBatch(frame: Frame): RecordBatch {
  const center = vectorFromArray([ZERO2], f32x2).data[0];
  const size = vectorFromArray([ONE2], f32x2).data[0];
  const theta = vectorFromArray([0], new Float32()).data[0];
  const model = vectorFromArray([frame.model], mat4).data[0];
  const eye = vectorFromArray([frame.eye], f32x3).data[0];
  const target = vectorFromArray([frame.target], f32x3).data[0];
  const up = vectorFromArray([frame.up], f32x3).data[0];
  const fovy = vectorFromArray([frame.fovy], new Float32()).data[0];
  const aspect = vectorFromArray([frame.aspect], new Float32()).data[0];
  if (!center || !size || !theta || !model || !eye || !target || !up || !fovy || !aspect) {
    throw new Error("params Arrow vector construction produced no data");
  }
  return new RecordBatch(
    paramsSchema,
    makeData({
      type: new Struct(paramsSchema.fields),
      length: 1,
      nullCount: 0,
      children: [center, size, theta, model, eye, target, up, fovy, aspect],
    }),
  );
}

async function run(): Promise<void> {
  await init({ module_or_path: wasmUrl });

  // Render resolution = the canvas drawing-buffer size (default 1024², matching
  // the dolly camera's aspect = 1.0); `?size=N` overrides it. Must run before
  // `create`, which reads width/height at creation time.
  const query = new URLSearchParams(window.location.search);
  const sizeParam = Number(query.get("size"));
  if (Number.isFinite(sizeParam) && sizeParam >= 16 && sizeParam <= 4096) {
    const size = Math.floor(sizeParam);
    canvasElement.width = size;
    canvasElement.height = size;
  }
  const fpsParam = Number(query.get("fps"));
  const fps = Number.isFinite(fpsParam) && fpsParam >= 1 && fpsParam <= 240 ? fpsParam : 24;

  const renderer = await CanvasRenderer.create(canvasElement);
  // Match the native `--texture --aabb --axes` scene: sample the bound atlas at
  // each vertex UV and overlay the mesh's AABB box + the world coordinate-axes.
  renderer.setTextured(true);
  renderer.setShowAabb(true);
  renderer.setShowAxes(true);

  statusElement.textContent = "textured bunny: loading mesh + atlas…";
  const [{ positions, uvs, indices }, texture, frames] = await Promise.all([
    fetch(bunnyUrl).then(async (response) => {
      if (!response.ok) {
        throw new Error(`failed to load bunny.obj: ${response.status}`);
      }
      return parseTexturedObj(await response.text());
    }),
    decodeTextureRgba(textureUrl),
    loadFrames(framesUrl),
  ]);
  if (positions.length === 0 || indices.length === 0) {
    throw new Error("bunny OBJ parsed to an empty mesh");
  }
  if (frames.length === 0) {
    throw new Error("dolly frame stream is empty");
  }

  // Deliver the leading [mesh][texture] tables before any params frame so the
  // renderer (built lazily on the first frame) binds the atlas as its albedo.
  renderer.pushIpc(meshStreamBytes(positions, uvs, indices));
  renderer.pushIpc(textureStreamBytes(texture.width, texture.height, texture.rgba));

  // Stream params frames one at a time on a single Arrow IPC sub-stream (a params
  // stream must stay one stream, so it is chunked rather than re-opened), cycling
  // the dolly frames forever at `fps` so the textured bunny keeps rotating.
  const writer = new RecordBatchStreamWriter({ compressionType: null });
  void (async () => {
    for await (const chunk of writer) {
      renderer.pushIpc(chunk);
    }
  })().catch((error: unknown) => {
    statusElement.textContent = `textured bunny: error — ${String(error)}`;
    throw error;
  });

  statusElement.textContent = `textured bunny: streaming (${texture.width}x${texture.height} atlas)`;
  let frame = 0;
  const step = () => {
    const current = frames[frame % frames.length];
    if (current) {
      writer.write(frameBatch(current));
    }
    frame += 1;
    statusElement.dataset.texturedFrames = String(frame);
    window.setTimeout(step, 1000 / fps);
  };
  window.setTimeout(step, 1000 / fps);
}

await run();

export {};
