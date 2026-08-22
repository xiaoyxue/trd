// Local test helper: serve a directory with CORS **and byte ranges**, so the
// browser editor can load a video *and* an annotation document from an origin
// other than the dev server (#264). Not part of the build — run it by hand:
//
//   bun web/gui-video-editing/serve-documents.ts [directory] [--port 8090] [--log]
//
// Four things a naive static server gets wrong here. A cross-origin fetch needs
// `Access-Control-Allow-Origin` (python's `http.server` sends none, which is why
// a document fetch fails with a bare TypeError rather than a status code);
// seeking needs `Range` — without a `206` a reader can only play straight
// through; every response has to be **streamed**, because the videos this serves
// are hundreds of gigabytes and materialising one response body is an
// out-of-memory crash rather than a slow request; and a connection this closes
// has to *say* it is closing (#326).
//
// That last one is why the native editor's `--video-url` could not be verified
// at all. Bun closes the socket when a request says `Connection: close`, but
// does not put `Connection: close` in the reply — so an HTTP/1.1 client is
// entitled to read the reply as persistent. ffmpeg does exactly that: it opens
// with `Range: bytes=0-`, then for the seek to `moov` writes its next request
// onto the socket Bun has already closed, reads a reset instead of a response,
// falls back to the stale connection, re-parses the same bytes, and repeats
// until it dies (`0xC0000005` on Windows, or a whole-file drain — 86 s for
// 1.48 GiB, unbounded for 218 GiB). Echoing the header costs one line and the
// same probe finishes in 0.11 s. A browser never saw it because `fetch` owns its
// own connections and mediabunny asks for bounded ranges.

import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { resolve } from "node:path";
import { Readable } from "node:stream";
import { fileURLToPath, pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const portFlag = args.indexOf("--port");
const port = Number(portFlag >= 0 ? args[portFlag + 1] : (process.env.PORT ?? 8090));
const log = args.includes("--log");
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

/// `--log`: one line per request, then the delivered bytes when its body ends —
/// the number §4.7 is actually asking for when it says opening must cost
/// megabytes, not gigabytes. Delivered, not requested: ffmpeg opens with
/// `bytes=0-` and abandons it, so the announced length says nothing.
let requests = 0;
let delivered = 0;
const mib = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(2)} MiB`;

/// A response body that reads the file as the client drains it. `Bun.file()`
/// slices materialise the whole range, which for a 200 GiB video means the
/// server grows to tens of gigabytes and is killed — so the byte count a
/// request names must never decide how much memory it costs.
function streamRange(id: number, path: string, start: number, end: number): ReadableStream {
  const file = createReadStream(path, { start, end });
  // `bytesRead`, not a `data` listener: attaching one would switch the stream to
  // flowing mode out from under `Readable.toWeb`.
  if (log) {
    file.on("close", () => {
      delivered += file.bytesRead;
      console.log(
        `  #${id} delivered ${mib(file.bytesRead)} (${requests} requests, ${mib(delivered)} total)`,
      );
    });
  }
  return Readable.toWeb(file) as ReadableStream;
}

/// Whether the client asked for the connection to end after this response.
///
/// Bun honours that request but does not announce it, and a reply that closes
/// without saying so is what breaks ffmpeg's seek (see the header comment).
function connectionHeader(request: Request): Record<string, string> {
  const close = (request.headers.get("connection") ?? "")
    .split(",")
    .some((token) => token.trim().toLowerCase() === "close");
  return close ? { Connection: "close" } : {};
}

Bun.serve({
  port,
  // Serving a range out of a huge file can outlast the default timeout, and a
  // request that times out mid-body looks to the reader like a truncated file.
  idleTimeout: 120,
  async fetch(request) {
    const id = ++requests;
    const name = decodeURIComponent(new URL(request.url).pathname.replace(/^\/+/, ""));
    // Echoing `Connection: close` is not cosmetic: Bun ends the socket when a
    // request asks it to, and a client told nothing reuses the dead connection.
    const base = { ...CORS, ...connectionHeader(request) };
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: base });
    }
    const path = name ? fileURLToPath(new URL(name, root)) : "";
    const info = path ? await stat(path).catch(() => undefined) : undefined;
    if (!info?.isFile()) {
      return new Response("not found", { status: 404, headers: base });
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
        headers: { ...base, "Content-Range": `bytes */${size}` },
      });
    }
    const last = Math.min(end, size - 1);
    const length = last - start + 1;
    const headers = {
      ...base,
      "Content-Type": "application/octet-stream",
      "Content-Length": String(match ? length : size),
      ...(match ? { "Content-Range": `bytes ${start}-${last}/${size}` } : {}),
    };
    const status = match ? 206 : 200;
    if (log) {
      console.log(
        `#${id} ${request.method} /${name} ${range ?? "(no range)"} -> ${status} ` +
          `bytes ${start}-${last}/${size}`,
      );
    }
    // A `HEAD` answers with the headers alone; giving it a body is what made
    // the length probe in `byte-source.ts` try to send the whole video.
    if (request.method === "HEAD") {
      return new Response(null, { status, headers });
    }
    return new Response(streamRange(id, path, start, last), { status, headers });
  },
});

console.log(
  `serving ${root.pathname} with CORS + ranges on http://localhost:${port}/` +
    (log ? " (--log: per-request bytes)" : ""),
);
