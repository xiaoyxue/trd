import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createVideoEditorApi, type VideoEditorBackend } from "./api.ts";

function backend() {
  const calls: string[] = [];
  let document: Uint8Array | undefined;
  const state = { playing: true, seconds: 4.5 };
  const host: VideoEditorBackend = {
    async loadArrow(bytes) {
      calls.push("load");
      if (bytes[0] === 255) throw new Error("invalid protocol input");
      document = bytes;
    },
    async resetState() {
      calls.push("reset");
      document = undefined;
    },
    async exportArrow() {
      calls.push("export");
      if (!document) throw new Error("no current Arrow document to export");
      return document.slice();
    },
    async seekToSeconds(seconds) {
      calls.push("seek");
      state.seconds = seconds;
    },
  };
  return { host, calls, state };
}

describe("external video-editor API", () => {
  test.skipIf(!process.env.TRD_TEST_API_PACKAGE)(
    "built package exports working JS and declarations",
    async () => {
      const root = resolve(import.meta.dir, "..");
      const pkg = await Bun.file(resolve(root, "package.json")).json();
      const entry = pkg.exports["."];
      const sdk = await import(pathToFileURL(resolve(root, entry.import)).href);
      const declarations = await Bun.file(resolve(root, entry.types)).text();
      expect(declarations).toContain("export interface VideoEditorApi");
      const { host } = backend();
      const api = sdk.createVideoEditorApi(host);
      await api.loadArrow(new Uint8Array([1, 2]));
      expect(await api.exportArrow()).toEqual(new Uint8Array([1, 2]));
      await api.resetState();
      await expect(api.exportArrow()).rejects.toThrow("no current Arrow");
    },
  );

  test("snapshots ordered byte parts before a queued load executes", async () => {
    const { host, calls } = backend();
    const api = createVideoEditorApi(host);
    const params = new Uint8Array([1, 2]);
    const meshes = new Uint8Array([3, 4]);
    const loading = api.loadArrow([params, meshes]);
    params.fill(9);
    meshes.fill(9);
    await loading;
    expect(await api.exportArrow()).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(calls).toEqual(["load", "export"]);
  });

  test("a single buffer is snapshotted too", async () => {
    const { host } = backend();
    const api = createVideoEditorApi(host);
    const bytes = new Uint8Array([1, 2]);
    const loading = api.loadArrow(bytes);
    bytes.fill(8);
    await loading;
    expect(await api.exportArrow()).toEqual(new Uint8Array([1, 2]));
  });

  test("reset, replacement and export wait for the prior load acknowledgement", async () => {
    const { host, calls, state } = backend();
    let acknowledge: () => void = () => {
      throw new Error("acknowledgement not installed");
    };
    const ready = new Promise<void>((resolve) => {
      acknowledge = resolve;
    });
    const load = host.loadArrow;
    let first = true;
    host.loadArrow = async (bytes) => {
      if (first) {
        first = false;
        calls.push("waiting");
        await ready;
      }
      await load(bytes);
    };
    const api = createVideoEditorApi(host);
    const a = api.loadArrow(new Uint8Array([1]));
    const reset = api.resetState();
    const b = api.loadArrow([new Uint8Array([2]), new Uint8Array([3])]);
    const exported = api.exportArrow();
    await Promise.resolve();
    expect(calls).toEqual(["waiting"]);
    acknowledge();
    await Promise.all([a, reset, b]);
    expect(await exported).toEqual(new Uint8Array([2, 3]));
    expect(calls).toEqual(["waiting", "load", "reset", "load", "export"]);
    expect(state).toEqual({ playing: true, seconds: 4.5 });
  });

  test("failed input leaves the backend document intact and does not poison the queue", async () => {
    const { host } = backend();
    const api = createVideoEditorApi(host);
    await api.loadArrow(new Uint8Array([1]));
    const rejected = api.loadArrow(new Uint8Array([255]));
    const exported = api.exportArrow();
    await expect(rejected).rejects.toThrow("invalid protocol");
    expect(await exported).toEqual(new Uint8Array([1]));
    await api.resetState();
    await expect(api.exportArrow()).rejects.toThrow("no current Arrow");
    await api.loadArrow(new Uint8Array([2]));
    expect(await api.exportArrow()).toEqual(new Uint8Array([2]));
  });

  test("empty input is rejected, not interpreted as reset", async () => {
    const { host, calls } = backend();
    const api = createVideoEditorApi(host);
    await expect(api.loadArrow([])).rejects.toThrow("complete Arrow document");
    await expect(api.loadArrow(new Uint8Array())).rejects.toThrow("nonzero");
    expect(calls).toEqual([]);
  });

  test("JavaScript callers receive explicit type errors", async () => {
    const { host, calls } = backend();
    const api = createVideoEditorApi(host);
    await expect(Reflect.apply(api.loadArrow, api, ["not Arrow"])).rejects.toThrow(
      "complete Arrow",
    );
    await expect(Reflect.apply(api.loadArrow, api, [[null]])).rejects.toThrow("Uint8Array");
    expect(calls).toEqual([]);
  });

  test("seeks are ordered with scene operations and reject invalid seconds", async () => {
    const { host, calls, state } = backend();
    const api = createVideoEditorApi(host);
    for (const seconds of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      await expect(api.seekToSeconds(seconds)).rejects.toThrow("finite and nonnegative");
    }
    await Promise.all([
      api.resetState(),
      api.loadArrow(new Uint8Array([1])),
      api.seekToSeconds(8.25),
    ]);
    expect(calls).toEqual(["reset", "load", "seek"]);
    expect(state).toEqual({ playing: true, seconds: 8.25 });
  });
});
