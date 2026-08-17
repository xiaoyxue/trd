import { afterEach, describe, expect, test } from "bun:test";

import { fileByteSource, urlByteSource } from "./byte-source.ts";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

interface Served {
  status?: number;
  headers?: Record<string, string>;
  body?: Uint8Array;
}

/// Replaces `fetch` and records what was asked for, so the range protocol can
/// be exercised without a server.
function serve(
  handler: (request: { url: string; method: string; range: string | null }) => Served,
) {
  const calls: { url: string; method: string; range: string | null }[] = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    const call = {
      url: String(input),
      method: init?.method ?? "GET",
      range: headers.get("range"),
    };
    calls.push(call);
    const served = handler(call);
    return new Response((served.body ?? new Uint8Array(0)) as unknown as BodyInit, {
      status: served.status ?? 206,
      headers: served.headers,
    });
  }) as typeof fetch;
  return calls;
}

/// A server that answers ranges correctly out of `bytes`.
function rangeServer(bytes: Uint8Array) {
  return serve(({ method, range }): Served => {
    const match = range?.match(/bytes=(\d+)-(\d+)/);
    if (method === "HEAD" || !match?.[1] || !match[2]) {
      return { status: 200, headers: { "content-length": String(bytes.byteLength) } };
    }
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), bytes.byteLength - 1);
    return {
      status: 206,
      headers: { "content-range": `bytes ${start}-${end}/${bytes.byteLength}` },
      body: bytes.slice(start, end + 1),
    };
  });
}

const payload = new Uint8Array(Array.from({ length: 256 }, (_, index) => index));

describe("fileByteSource", () => {
  test("reads a slice and counts only what it transferred", async () => {
    const source = fileByteSource(new File([payload], "clip.mp4"));
    expect(source.size).toBe(256);
    expect(source.label).toBe("clip.mp4");

    const first = new Uint8Array(await source.read(10, 4));
    expect([...first]).toEqual([10, 11, 12, 13]);
    expect(source.bytesRead).toBe(4);
  });

  test("clamps a read that runs past the end, and returns nothing beyond it", async () => {
    const source = fileByteSource(new File([payload], "clip.mp4"));

    // A short read at the end is normal, not an error: the last chunk of a file
    // is rarely a whole chunk.
    expect((await source.read(250, 64)).byteLength).toBe(6);
    expect((await source.read(256, 16)).byteLength).toBe(0);
    expect((await source.read(9999, 16)).byteLength).toBe(0);
  });
});

describe("urlByteSource", () => {
  test("takes the length from Content-Range and then reads ranges", async () => {
    const calls = rangeServer(payload);
    const source = await urlByteSource("https://example.test/clip.mp4");

    expect(source.size).toBe(256);
    // The opening probe asks for a single byte rather than the file.
    expect(calls[0]?.range).toBe("bytes=0-0");

    const bytes = new Uint8Array(await source.read(32, 8));
    expect([...bytes]).toEqual([32, 33, 34, 35, 36, 37, 38, 39]);
    expect(calls[1]?.range).toBe("bytes=32-39");
  });

  test("falls back to a HEAD when Content-Range is not exposed", async () => {
    // Cross-origin, `Content-Range` is not safelisted, so a server that serves
    // ranges can still hide the length. `Content-Length` on a HEAD is
    // safelisted, which is why that fallback exists at all.
    const calls = serve(({ method }) =>
      method === "HEAD"
        ? { status: 200, headers: { "content-length": "4096" } }
        : { status: 206, body: new Uint8Array(1) },
    );
    const source = await urlByteSource("https://example.test/clip.mp4");

    expect(source.size).toBe(4096);
    expect(calls.map((call) => call.method)).toEqual(["GET", "HEAD"]);
  });

  test("rejects a 200 rather than downloading a whole large file", async () => {
    // A server that ignores `Range` answers 200 with everything. Treating that
    // as a slow path would turn one seek into a multi-gigabyte download.
    serve(() => ({
      status: 200,
      headers: { "content-length": String(512 * 1024 * 1024) },
      body: new Uint8Array(0),
    }));

    await expect(urlByteSource("https://example.test/huge.mp4")).rejects.toThrow(
      /does not support Range/,
    );
  });

  test("downloads a small file whole when the server refuses ranges", async () => {
    serve(() => ({
      status: 200,
      headers: { "content-length": String(payload.byteLength) },
      body: payload,
    }));
    const source = await urlByteSource("https://example.test/small.mp4");

    expect(source.size).toBe(256);
    expect([...new Uint8Array(await source.read(2, 3))]).toEqual([2, 3, 4]);
  });

  test("refuses a range answered from the wrong offset", async () => {
    // Legal to serve fewer bytes than asked for; not legal to serve *other*
    // bytes. The demuxer is told where the buffer belongs, so bytes from
    // elsewhere are parsed as if they were the requested ones — silent
    // corruption unless the reported start is checked.
    let first = true;
    serve(() => {
      if (first) {
        first = false;
        return {
          status: 206,
          headers: { "content-range": "bytes 0-0/256" },
          body: payload.slice(0, 1),
        };
      }
      return {
        status: 206,
        headers: { "content-range": "bytes 999-1006/2048" },
        body: payload.slice(0, 8),
      };
    });
    const source = await urlByteSource("https://example.test/clip.mp4");

    await expect(source.read(32, 8)).rejects.toThrow(/was answered from byte 999/);
  });

  test("accepts a short range, which a server may legally return", async () => {
    let first = true;
    serve(() => {
      if (first) {
        first = false;
        return {
          status: 206,
          headers: { "content-range": "bytes 0-0/256" },
          body: payload.slice(0, 1),
        };
      }
      return {
        status: 206,
        headers: { "content-range": "bytes 32-35/256" },
        body: payload.slice(32, 36),
      };
    });
    const source = await urlByteSource("https://example.test/clip.mp4");

    expect((await source.read(32, 64)).byteLength).toBe(4);
  });

  test("reports a refused request as a CORS problem rather than a status", async () => {
    // A `fetch` rejection is the browser refusing to make the request, so there
    // is no status to report — almost always a missing CORS header.
    globalThis.fetch = (() =>
      Promise.reject(new TypeError("Failed to fetch"))) as unknown as typeof fetch;

    await expect(urlByteSource("https://example.test/clip.mp4")).rejects.toThrow(
      /Access-Control-Allow-Origin/,
    );
  });

  test("refuses a source whose length cannot be determined", async () => {
    serve(({ method }) =>
      method === "HEAD" ? { status: 200 } : { status: 206, body: new Uint8Array(1) },
    );

    await expect(urlByteSource("https://example.test/clip.mp4")).rejects.toThrow(
      /cannot determine the length/,
    );
  });

  test("counts bytes transferred, which is what proves a seek stayed cheap", async () => {
    rangeServer(payload);
    const source = await urlByteSource("https://example.test/clip.mp4");
    const opening = source.bytesRead;

    await source.read(0, 16);
    await source.read(100, 16);

    expect(source.bytesRead - opening).toBe(32);
  });

  test("returns nothing for a read at or past the end without asking the server", async () => {
    const calls = rangeServer(payload);
    const source = await urlByteSource("https://example.test/clip.mp4");
    const before = calls.length;

    expect((await source.read(256, 16)).byteLength).toBe(0);
    expect((await source.read(300, 16)).byteLength).toBe(0);
    expect(calls.length).toBe(before);
  });
});
