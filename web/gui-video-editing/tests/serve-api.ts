import { createReadStream } from "node:fs";
import { Readable } from "node:stream";
import api from "./api.html";

function rangedBody(path: string, start: number, end: number): ReadableStream<Uint8Array> {
  const reader = Readable.toWeb(createReadStream(path, { start, end })).getReader();
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await reader.read();
      if (next.done) controller.close();
      else controller.enqueue(next.value);
    },
    cancel: (reason) => reader.cancel(reason),
  });
}

const files = new Map<string, string>();
for (const [route, variable] of [
  ["/input/params.arrow", "TRD_API_PARAMS"],
  ["/input/single.arrow", "TRD_API_SINGLE"],
  ["/input/multiple.arrow", "TRD_API_MULTIPLE"],
  ["/video.mp4", "TRD_API_VIDEO"],
] as const) {
  const path = process.env[variable];
  if (!path) throw new Error(`Set ${variable} to the matching external test input`);
  files.set(route, path);
}
Bun.serve({
  hostname: "127.0.0.1",
  port: Number(process.env.BUN_PORT ?? 18370),
  development: false,
  idleTimeout: 120,
  routes: { "/": api },
  async fetch(request) {
    const path = files.get(new URL(request.url).pathname);
    if (!path) return new Response("not found", { status: 404 });
    const file = Bun.file(path);
    if (!(await file.exists())) return new Response("not found", { status: 404 });
    const range = request.headers.get("range")?.match(/^bytes=(\d+)-(\d*)$/);
    const start = range ? Number(range[1]) : 0;
    const end = Math.min(range?.[2] ? Number(range[2]) : file.size - 1, file.size - 1);
    if (start > end || start >= file.size) return new Response(null, { status: 416 });
    const headers = {
      "Content-Type": file.type,
      "Content-Length": String(end - start + 1),
      "Accept-Ranges": "bytes",
      ...(range ? { "Content-Range": `bytes ${start}-${end}/${file.size}` } : {}),
    };
    return new Response(request.method === "HEAD" ? null : rangedBody(path, start, end), {
      status: range ? 206 : 200,
      headers,
    });
  },
});
console.log("API E2E: /?document=none&video=/video.mp4");
