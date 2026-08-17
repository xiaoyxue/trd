// Local test helper: serve a directory with CORS **and byte ranges**, so the
// browser editor can load a video *and* an annotation document from an origin
// other than the dev server (#264). Not part of the build — run it by hand:
//
//   bun web/gui-video-editing/serve-documents.ts [directory] [--port 8090]
//
// Three things a naive static server gets wrong here. A cross-origin fetch needs
// `Access-Control-Allow-Origin` (python's `http.server` sends none, which is why
// a document fetch fails with a bare TypeError rather than a status code);
// seeking needs `Range` — without a `206` a reader can only play straight
// through; and every response has to be **streamed**, because the videos this
// serves are hundreds of gigabytes and materialising one response body is an
// out-of-memory crash rather than a slow request.

import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve } from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const portFlag = args.indexOf("--port");
const port = Number(portFlag >= 0 ? args[portFlag + 1] : (process.env.PORT ?? 8090));
const rootArg = args.find((value, index) => !value.startsWith("--") && index !== portFlag + 1);
// `new URL(dir, import.meta.url)` mis-reads an absolute Windows path — `D:` is
// parsed as a URL scheme — so resolve through the filesystem instead. The
// trailing `/` is what makes it a *base directory*: without it `new URL(name,
// root)` would replace the last segment and serve the parent instead. It has to
// be a `/` rather than the platform separator, because by then this is a URL.
const root = rootArg
  ? new URL(`${pathToFileURL(resolve(rootArg)).href}/`)
  : new URL("./data/", import.meta.url);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "range",
  "Access-Control-Expose-Headers": "content-length, content-range, accept-ranges",
  "Accept-Ranges": "bytes",
};

/// A response body that reads the file as the client drains it. `Bun.file()`
/// slices materialise the whole range, which for a 200 GiB video means the
/// server grows to tens of gigabytes and is killed — so the byte count a
/// request names must never decide how much memory it costs.
function streamRange(path: string, start: number, end: number): ReadableStream {
  return Readable.toWeb(createReadStream(path, { start, end })) as ReadableStream;
}

Bun.serve({
  port,
  // Serving a range out of a huge file can outlast the default timeout, and a
  // request that times out mid-body looks to the reader like a truncated file.
  idleTimeout: 120,
  async fetch(request) {
    const name = decodeURIComponent(new URL(request.url).pathname.replace(/^\/+/, ""));
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }
    const path = name ? fileURLToPath(new URL(name, root)) : "";
    const info = path ? await stat(path).catch(() => undefined) : undefined;
    if (!info?.isFile()) {
      return new Response("not found", { status: 404, headers: CORS });
    }
    const size = info.size;
    const range = request.headers.get("range");
    const match = range?.match(/bytes=(\d*)-(\d*)/);
    // `bytes=start-` (no end) is open-ended and legal, and it is exactly the
    // shape that used to ask for the entire remainder of the file at once.
    const start = match ? (match[1] ? Number(match[1]) : 0) : 0;
    const end = match?.[2] ? Number(match[2]) : size - 1;
    if (match && (start >= size || end < start)) {
      return new Response("range not satisfiable", {
        status: 416,
        headers: { ...CORS, "Content-Range": `bytes */${size}` },
      });
    }
    const last = Math.min(end, size - 1);
    const length = last - start + 1;
    const headers = {
      ...CORS,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(match ? length : size),
      ...(match ? { "Content-Range": `bytes ${start}-${last}/${size}` } : {}),
    };
    // A `HEAD` answers with the headers alone; giving it a body is what made
    // the length probe in `byte-source.ts` try to send the whole video.
    if (request.method === "HEAD") {
      return new Response(null, { status: match ? 206 : 200, headers });
    }
    return new Response(streamRange(path, start, last), {
      status: match ? 206 : 200,
      headers,
    });
  },
});

console.log(`serving ${root.pathname} with CORS + ranges on http://localhost:${port}/`);
