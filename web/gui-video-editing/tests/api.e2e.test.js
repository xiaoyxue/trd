import { describe, expect, test } from "bun:test";
import { resolve } from "node:path";

const endpoint = process.env.TRD_API_CDP;
const url = process.env.TRD_API_E2E_URL;

async function browser(work) {
  const pages = await (await fetch(`${endpoint}/json/list`)).json();
  const page = pages.find((entry) => entry.type === "page" && entry.url.startsWith(url));
  expect(page).toBeDefined();
  const socket = new WebSocket(page.webSocketDebuggerUrl);
  const pending = new Map();
  let id = 0;
  try {
    await new Promise((resolve, reject) => {
      socket.addEventListener("open", resolve, { once: true });
      socket.addEventListener("error", reject, { once: true });
    });
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const request = pending.get(message.id);
      if (!request) return;
      pending.delete(message.id);
      clearTimeout(request.timer);
      if (message.error) request.reject(new Error(JSON.stringify(message.error)));
      else request.resolve(message.result);
    });
    async function evaluate(expression) {
      const response = await new Promise((resolve, reject) => {
        const seq = ++id;
        const timer = setTimeout(() => {
          pending.delete(seq);
          reject(new Error("browser API E2E timed out"));
        }, 90_000);
        pending.set(seq, { resolve, reject, timer });
        socket.send(
          JSON.stringify({
            id: seq,
            method: "Runtime.evaluate",
            params: { expression, awaitPromise: true, returnByValue: true },
          }),
        );
      });
      if (response.exceptionDetails) throw new Error(JSON.stringify(response.exceptionDetails));
      return response.result.value;
    }
    return await work(evaluate);
  } finally {
    socket.close();
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(new Error("browser case closed"));
    }
  }
}

describe.skipIf(!endpoint || !url)("external TS and real WASM API E2E", () => {
  test("integration: current input, reset, exported edits and exact seeks", async () => {
    const result = await browser(async (evaluate) => {
      await evaluate("window.trdVideoEditorReady.then(() => true)");
      return evaluate("window.trdApiE2e.run()");
    });
    expect(Object.keys(result.exports)).toEqual(["params", "single", "multiple", "edited"]);
    expect(result.media.frame).toBe(24);
    expect(result.media.sourceChanges).toBe(1);
    for (const method of ["loadArrow", "resetState", "exportArrow", "seekToSeconds"]) {
      expect(result.calls[method]).toBeGreaterThan(0);
    }
    if (process.env.TRD_API_RESULTS) {
      const directory = resolve(process.env.TRD_API_RESULTS);
      await Bun.write(resolve(directory, "integration.json"), JSON.stringify(result, null, 2));
    }
  }, 120_000);

  test.skipIf(process.env.TRD_API_PLAYING !== "1")(
    "playing: GUI play continues across WASM reset and TS reload",
    async () => {
      const result = await browser((evaluate) => evaluate("window.trdApiE2e.resetPlaying()"));
      expect(result.after.playing).toBe(true);
      expect(result.after.sourceChanges).toBe(result.before.sourceChanges);
      expect(result.after.seconds).toBeGreaterThan(result.before.seconds);
      if (process.env.TRD_API_RESULTS) {
        await Bun.write(
          resolve(process.env.TRD_API_RESULTS, "playing.json"),
          JSON.stringify(result, null, 2),
        );
      }
    },
    30_000,
  );
});
