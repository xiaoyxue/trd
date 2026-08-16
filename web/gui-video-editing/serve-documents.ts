// Local test helper: serve a directory with CORS **and byte ranges**, so the
// browser editor can load a video *and* an annotation document from an origin
// other than the dev server (#264). Not part of the build — run it by hand:
//
//   bun web/gui-video-editing/serve-documents.ts [directory] [--port 8090]
//
// Two things a naive static server gets wrong here. A cross-origin fetch needs
// `Access-Control-Allow-Origin` (python's `http.server` sends none, which is why
// a document fetch fails with a bare TypeError rather than a status code), and
// `<video>` seeking needs `Range` — without a `206` the element can only play
// straight through.
import { file } from "bun";

const args = process.argv.slice(2);
const portFlag = args.indexOf("--port");
const port = Number(portFlag >= 0 ? args[portFlag + 1] : (process.env.PORT ?? 8090));
const rootArg = args.find((value, index) => !value.startsWith("--") && index !== portFlag + 1);
const root = new URL(rootArg ? `${rootArg.replace(/\\/g, "/")}/` : "./data/", import.meta.url);

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "range",
  "Access-Control-Expose-Headers": "content-length, content-range, accept-ranges",
  "Accept-Ranges": "bytes",
};

Bun.serve({
  port,
  async fetch(request) {
    const name = decodeURIComponent(new URL(request.url).pathname.replace(/^\/+/, ""));
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS });
    }
    const candidate = file(new URL(name, root));
    if (!name || !(await candidate.exists())) {
      return new Response("not found", { status: 404, headers: CORS });
    }
    const size = candidate.size;
    const range = request.headers.get("range");
    const match = range?.match(/bytes=(\d*)-(\d*)/);
    if (match) {
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Number(match[2]) : size - 1;
      return new Response(candidate.slice(start, end + 1), {
        status: 206,
        headers: { ...CORS, "Content-Range": `bytes ${start}-${end}/${size}` },
      });
    }
    return new Response(candidate, { headers: CORS });
  },
});

console.log(`serving ${root.pathname} with CORS + ranges on http://localhost:${port}/`);
