import { afterEach, describe, expect, test } from "bun:test";

import { fetchWithRetry, retryDelaySeconds } from "./fetch-retry.ts";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("retryDelaySeconds", () => {
  test("backs off, then gives up so a standing failure still reports", () => {
    expect(retryDelaySeconds(1)).toBe(0.25);
    expect(retryDelaySeconds(2)).toBe(0.5);
    expect(retryDelaySeconds(3)).toBe(1);
    // The point of the bound: a genuine CORS misconfiguration cannot retry
    // forever, it just reports ~1.75s later than it used to.
    expect(retryDelaySeconds(4)).toBeNull();
  });
});

describe("fetchWithRetry", () => {
  test("recovers from a transient rejection instead of failing the load", async () => {
    let attempts = 0;
    globalThis.fetch = (async (_input: string | URL | Request, _init?: RequestInit) => {
      attempts += 1;
      if (attempts === 1) {
        // What a cross-origin blip looks like: the browser refused to make the
        // request, so there is no status — indistinguishable from CORS, which
        // is why mediabunny's default gave up here.
        throw new TypeError("Failed to fetch");
      }
      return new Response(new Uint8Array([1]) as unknown as BodyInit, { status: 206 });
    }) as typeof fetch;

    const response = await fetchWithRetry("http://example.test/clip.mp4");
    expect(response.status).toBe(206);
    expect(attempts).toBe(2);
  });

  test("gives up once the retries are spent", async () => {
    let attempts = 0;
    // Annotated: a stub that only ever throws infers `Promise<never>`, which
    // does not overlap `typeof fetch`.
    globalThis.fetch = (async (
      _input: string | URL | Request,
      _init?: RequestInit,
    ): Promise<Response> => {
      attempts += 1;
      throw new TypeError("Failed to fetch");
    }) as typeof fetch;

    await expect(fetchWithRetry("http://example.test/clip.mp4")).rejects.toThrow("Failed to fetch");
    expect(attempts).toBe(4); // the first try plus MAX_FETCH_RETRIES
  });

  test("an answered request is the server's verdict, not a blip", async () => {
    let attempts = 0;
    globalThis.fetch = (async (_input: string | URL | Request, _init?: RequestInit) => {
      attempts += 1;
      return new Response(null, { status: 404, statusText: "Not Found" });
    }) as typeof fetch;

    const response = await fetchWithRetry("http://example.test/missing.mp4");
    expect(response.status).toBe(404);
    expect(attempts).toBe(1);
  });
});
