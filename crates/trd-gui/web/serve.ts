// Dev/serve entry for the browser trd-gui viewer. `bun ./index.html`'s built-in
// server is SPA-style (unknown routes fall back to index.html), so a fetched
// `?mesh=`/`?texture=` asset would return HTML instead of the file. This uses
// `Bun.serve` with an HTML route (bundles index.html + src/main.ts + the wasm
// asset) plus a static handler that serves:
//   - files under this `web/` folder (e.g. the built pkg), and
//   - the repo's real `assets/` directory, so `?mesh=/assets/meshes/bunny.obj`
//     resolves to the same file the native `--mesh` reads — no copies.
import { existsSync } from "node:fs";
import index from "./index.html";

const port = Number(process.env.BUN_PORT ?? 8080);
const REPO_ROOT = new URL("../../../", import.meta.url);

const server = Bun.serve({
  port,
  development: true,
  routes: { "/": index },
  async fetch(req) {
    const { pathname } = new URL(req.url);
    if (pathname === "/") {
      return new Response("not found", { status: 404 });
    }

    // 1. Files under web/ (e.g. the built pkg/), served from the cwd.
    const local = Bun.file(`.${pathname}`);
    if (await local.exists()) {
      return new Response(local);
    }

    // 2. The repo's real assets: `/assets/...` maps to `<repo>/assets/...`, so the
    //    browser fetches the same OBJ/texture the native viewer reads. Scoped to
    //    `/assets/` so the dev server doesn't expose the whole repo.
    if (pathname.startsWith("/assets/")) {
      const repoUrl = new URL(`.${pathname}`, REPO_ROOT);
      if (existsSync(repoUrl)) {
        return new Response(Bun.file(repoUrl));
      }
    }

    return new Response("not found", { status: 404 });
  },
});

console.log(`trd-gui web on ${server.url}`);
