import { describe, expect, test } from "bun:test";

import type { ByteSource } from "./byte-source.ts";
import { locateMoov } from "./mp4-video.ts";

/// A `ByteSource` over bytes already in hand, counting reads so a test can
/// assert that finding `moov` never touched the `mdat` between the boxes.
function memorySource(bytes: Uint8Array, label = "test.mp4"): ByteSource & { reads: number } {
  let bytesRead = 0;
  let reads = 0;
  return {
    label,
    get size() {
      return bytes.byteLength;
    },
    get bytesRead() {
      return bytesRead;
    },
    get reads() {
      return reads;
    },
    async read(offset: number, length: number): Promise<ArrayBuffer> {
      reads += 1;
      const end = Math.min(offset + length, bytes.byteLength);
      if (end <= offset) {
        return new ArrayBuffer(0);
      }
      const slice = bytes.slice(offset, end);
      bytesRead += slice.byteLength;
      return slice.buffer as ArrayBuffer;
    },
  };
}

/// One top-level box header. `size` of 1 writes the 64-bit form, 0 means "runs
/// to the end of the file" — both are shapes real files use and neither is
/// reachable by writing a plain 32-bit header.
function boxHeader(kind: string, size: number, largesize?: number): Uint8Array {
  const header = new Uint8Array(largesize === undefined ? 8 : 16);
  const view = new DataView(header.buffer);
  view.setUint32(0, size);
  for (let index = 0; index < 4; index += 1) {
    header[4 + index] = kind.charCodeAt(index);
  }
  if (largesize !== undefined) {
    view.setBigUint64(8, BigInt(largesize));
  }
  return header;
}

/// Assembles a file from `[kind, totalSize, use64Bit?]` boxes, padding each to
/// its declared size so offsets are real.
function file(boxes: [string, number, boolean?][]): Uint8Array {
  const total = boxes.reduce((sum, [, size]) => sum + size, 0);
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const [kind, size, wide] of boxes) {
    const header = wide ? boxHeader(kind, 1, size) : boxHeader(kind, size);
    bytes.set(header, offset);
    offset += size;
  }
  return bytes;
}

describe("locateMoov", () => {
  test("finds a moov that follows ftyp, the fragmented-file layout", async () => {
    const bytes = file([
      ["ftyp", 32],
      ["moov", 1024],
      ["mdat", 4096],
    ]);
    const source = memorySource(bytes);
    const head = await source.read(0, 512);

    expect(await locateMoov(source, head)).toEqual({ offset: 32, size: 1024 });
  });

  test("finds a moov at the end, behind a 64-bit mdat, without reading it", async () => {
    // The 218 GiB recording's layout in miniature: `mdat` is too large for a
    // 32-bit size, so it declares 1 and carries a `largesize`. Getting this
    // wrong walks into the middle of the payload.
    const bytes = file([
      ["ftyp", 40],
      ["mdat", 100_000, true],
      ["moov", 2048],
    ]);
    const source = memorySource(bytes);
    const head = await source.read(0, 4096);
    const before = source.bytesRead;

    expect(await locateMoov(source, head)).toEqual({ offset: 40 + 100_000, size: 2048 });
    // Only box headers past the head read: never the payload in between.
    expect(source.bytesRead - before).toBeLessThan(64);
  });

  test("steps over a box whose size runs to the end of the file", async () => {
    const bytes = file([
      ["ftyp", 32],
      ["moov", 512],
    ]);
    // Rewrite `moov`'s size as 0 — "to end of file" — and append nothing else,
    // so the walk has to resolve it against the source length.
    new DataView(bytes.buffer).setUint32(32, 0);
    const source = memorySource(bytes);
    const head = await source.read(0, 64);

    expect(await locateMoov(source, head)).toEqual({ offset: 32, size: 512 });
  });

  test("keeps walking past boxes that come before moov", async () => {
    const bytes = file([
      ["ftyp", 32],
      ["free", 64],
      ["skip", 128],
      ["moov", 256],
      ["mfra", 48],
    ]);
    const source = memorySource(bytes);
    const head = await source.read(0, 1024);

    expect(await locateMoov(source, head)).toEqual({ offset: 32 + 64 + 128, size: 256 });
  });

  test("refuses a file with no moov rather than reporting one", async () => {
    const bytes = file([
      ["ftyp", 32],
      ["mdat", 512],
    ]);
    const source = memorySource(bytes);

    expect(await locateMoov(source, await source.read(0, 64))).toBeUndefined();
  });

  test("refuses a box whose declared size runs past the file", async () => {
    const bytes = file([["ftyp", 32]]);
    // 32 bytes of file claiming a 4 KiB box: truncated or corrupt, and stepping
    // by it would land outside the source.
    new DataView(bytes.buffer).setUint32(0, 4096);
    const source = memorySource(bytes);

    expect(await locateMoov(source, await source.read(0, 32))).toBeUndefined();
  });

  test("refuses a box smaller than its own header", async () => {
    const bytes = file([["ftyp", 32]]);
    // A size below 8 cannot even contain the size and type it just declared;
    // accepting it would advance the walk by less than a header, forever.
    new DataView(bytes.buffer).setUint32(0, 4);
    const source = memorySource(bytes);

    expect(await locateMoov(source, await source.read(0, 32))).toBeUndefined();
  });

  test("reads headers itself once the walk passes the end of the head buffer", async () => {
    const bytes = file([
      ["ftyp", 32],
      ["mdat", 200_000],
      ["moov", 512],
    ]);
    const source = memorySource(bytes);
    // A head far too small to reach `moov`: the walk has to fetch the header at
    // each hop, which is the path a real file with a trailing `moov` takes.
    const head = await source.read(0, 64);
    const readsBefore = source.reads;

    expect(await locateMoov(source, head)).toEqual({ offset: 32 + 200_000, size: 512 });
    expect(source.reads).toBeGreaterThan(readsBefore);
  });

  test("stops at a truncated header instead of reading past the end", async () => {
    // Four bytes is less than a header; the walk must not construct a view over
    // it or step by whatever it happens to read.
    const bytes = new Uint8Array(4);
    const source = memorySource(bytes);

    expect(await locateMoov(source, await source.read(0, 4))).toBeUndefined();
  });
});
