import type { DataType, Vector } from "apache-arrow";
import {
  Field,
  FixedSizeList,
  Float32,
  makeData,
  RecordBatch,
  RecordBatchStreamWriter,
  Schema,
  Struct,
  tableFromIPC,
  vectorFromArray,
} from "apache-arrow";
import init, { ArrowRenderer } from "trd-wasm";
import wasmUrl from "trd-wasm/trd_wasm_bg.wasm" with { type: "file" };

const width = 8;
const height = 4;
const pixels = width * height;

interface ArrayLikeArrowValue {
  toArray(): unknown;
}

function isArrayLikeArrowValue(value: unknown): value is ArrayLikeArrowValue {
  return (
    typeof value === "object" &&
    value !== null &&
    "toArray" in value &&
    typeof value.toArray === "function"
  );
}

function hasShape(value: unknown): value is { shape: readonly unknown[] } {
  return (
    typeof value === "object" && value !== null && "shape" in value && Array.isArray(value.shape)
  );
}

function concat(chunks: readonly Uint8Array[]): Uint8Array {
  const length = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  const bytes = new Uint8Array(length);
  let offset = 0;

  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return bytes;
}

const item = new Field("item", new Float32(), false);
const vec2 = new FixedSizeList(2, item);
const schema = new Schema(
  [
    new Field("center", vec2, false),
    new Field("size", vec2, false),
    new Field("theta", new Float32(), false),
  ],
  new Map([["trd.protocol.version", "0.0.1"]]),
);

function batch(center: number[][], size: number[][], theta: number[]): RecordBatch {
  if (center.length !== size.length || size.length !== theta.length) {
    throw new Error("input Arrow columns have mismatched lengths");
  }

  const vectors = [
    vectorFromArray(center, vec2),
    vectorFromArray(size, vec2),
    vectorFromArray(theta, new Float32()),
  ];

  const children = vectors.map((vector) => {
    const data = vector.data[0];
    if (data === undefined) {
      throw new Error("input Arrow vector has no first data chunk");
    }
    return data;
  });

  return new RecordBatch(
    schema,
    makeData({
      type: new Struct(schema.fields),
      length: theta.length,
      nullCount: 0,
      children,
    }),
  );
}

function fixedSizeListStorage(type: DataType): FixedSizeList {
  if (type instanceof FixedSizeList) {
    return type;
  }

  const storageType: unknown = Reflect.get(type, "storageType");
  if (storageType instanceof FixedSizeList) {
    return storageType;
  }

  throw new Error("output field is not fixed-size-list storage");
}

function tensorBytes(column: Vector, row: number): Uint8Array {
  const value = column.get(row);
  if (!isArrayLikeArrowValue(value)) {
    throw new Error("fixed-shape tensor row has no Arrow toArray method");
  }

  const bytes = value.toArray();
  if (!(bytes instanceof Uint8Array)) {
    throw new Error("fixed-shape tensor row is not Uint8Array");
  }

  return bytes;
}

function assertTensorField(field: Field): void {
  const storage = fixedSizeListStorage(field.type);

  if (storage.listSize !== pixels) {
    throw new Error(`${field.name} list size is ${storage.listSize}, expected ${pixels}`);
  }

  if (field.metadata.get("ARROW:extension:name") !== "arrow.fixed_shape_tensor") {
    throw new Error(`${field.name} lacks fixed_shape_tensor metadata`);
  }

  const rawMetadata = field.metadata.get("ARROW:extension:metadata");
  if (rawMetadata === undefined) {
    throw new Error(`${field.name} lacks tensor extension metadata`);
  }

  const metadata: unknown = JSON.parse(rawMetadata);
  if (
    !hasShape(metadata) ||
    metadata.shape.length !== 2 ||
    metadata.shape[0] !== height ||
    metadata.shape[1] !== width
  ) {
    throw new Error(`${field.name} tensor shape is not [${height}, ${width}]`);
  }
}

const first = batch([[0, 0]], [[0.75, 0.75]], [0]);
const second = batch(
  [
    [-0.2, 0.1],
    [0.2, -0.1],
  ],
  [
    [0.75, 0.75],
    [0.75, 0.75],
  ],
  [0.3, 0.6],
);

const input = await RecordBatchStreamWriter.writeAll([first, second]).toUint8Array();

await init({ module_or_path: wasmUrl });

const renderer = await ArrowRenderer.create(width, height);
const outputChunks: Uint8Array[] = [];

for (const chunk of [input.subarray(0, 5), input.subarray(5, 37), input.subarray(37)]) {
  outputChunks.push(await renderer.pushIpc(chunk));
}
outputChunks.push(renderer.finish());

let rejectedAfterFinish = false;
try {
  await renderer.pushIpc(new Uint8Array());
} catch (error) {
  rejectedAfterFinish = error instanceof Error;
}

if (!rejectedAfterFinish) {
  throw new Error("pushIpc after finish did not reject with Error");
}

const table = tableFromIPC(concat(outputChunks));

if (table.schema.metadata.get("trd.protocol.version") !== "0.0.1") {
  throw new Error("output protocol version is not 0.0.1");
}

if (table.batches.length !== 2) {
  throw new Error(`output batch count is ${table.batches.length}, expected 2`);
}

if (table.batches[0]?.numRows !== 1 || table.batches[1]?.numRows !== 2) {
  throw new Error("output batch row counts are not [1, 2]");
}

for (const name of ["r", "g", "b", "a"]) {
  const field = table.schema.fields.find((candidate) => candidate.name === name);
  if (field === undefined) {
    throw new Error(`missing output field ${name}`);
  }
  assertTensorField(field);
}

const r = table.getChild("r");
const g = table.getChild("g");
const b = table.getChild("b");
const a = table.getChild("a");

if (r === null || g === null || b === null || a === null) {
  throw new Error("missing output channel vector");
}

for (const channel of [r, g, b, a]) {
  for (let row = 0; row < 3; row += 1) {
    if (tensorBytes(channel, row).byteLength !== pixels) {
      throw new Error("output tensor pixel buffer has incorrect length");
    }
  }
}

if (tensorBytes(r, 0)[0] !== 0 || tensorBytes(g, 0)[0] !== 0 || tensorBytes(b, 0)[0] !== 0) {
  throw new Error("top-left RGB output pixel is not black");
}

if (tensorBytes(a, 0)[0] !== 255) {
  throw new Error("top-left alpha output pixel is not opaque");
}

document.body.dataset.arrowSmoke = "pass";
document.body.textContent = "PASS: ArrowRenderer Arrow IPC roundtrip";
