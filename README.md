# trd

A tile (relational) oriented renderer prototype, built on Rust + wgpu.

The Rust/wgpu core (`trd-core`) is the single rendering core; it runs natively
(headless CLI) and in the browser (compiled to wasm). JavaScript/TypeScript is a
thin bootstrap wrapper only — no WebGPU API is called from JS.

## Layout

- `crates/trd-core` — unified Rust/wgpu render core:
  - `render.rs` + `triangle.wgsl` — cross-platform parametric triangle renderer
  - `stream.rs` — native Arrow protocol, persistent GPU batch renderer, and
    Arrow IPC stdin/stdout pipeline
- `crates/trd-cli` — thin native headless CLI; Arrow IPC stdin → Arrow IPC stdout
- `crates/trd-app` — native interactive window (winit + a live wgpu surface)
- `crates/trd-wasm` — `wasm-bindgen` entry point; packaged as the `trd-wasm`
  npm library via `wasm-pack` (output in `crates/trd-wasm/pkg`, gitignored)
- `web/` — bun-managed thin TypeScript wrapper that consumes the `trd-wasm`
  package
- `examples/` — runnable JSONL animation example and `render.sh` wrapper
- `scripts/encode.py` — Arrow tensor stream → ffmpeg GIF/WebP adapter

## Development

The flake is the build system. Real, reproducible outputs:

```sh
nix build .#trd-cli   # native CLI (Arrow stream filter), wrapped with Vulkan/GL libs
nix build .#trd-wasm  # wasm-bindgen JS/TS library package
nix build .#web       # bun-bundled, HTTP-servable dist/ (also `nix build`)
nix run   .#trd -- --width 256 --height 256   # Arrow frames on stdin -> images on stdout
nix run   .#web                               # serve dist/ (PORT defaults to 8080)
nix flake check       # all gates: fmt, clippy (native+wasm32), test, tsc, biome
```

For fast local iteration, enter the dev shell (pinned Rust toolchain, `bun`,
`wasm-bindgen` / `wasm-pack`, `biome`, `typescript`, and Vulkan) and use plain
`cargo` / `bun`:

```sh
nix develop
```

> `nix build` and `nix flake check` only see git-tracked files — `git add` new
> files before building.

All commands below assume the dev shell is active. DuckDB is intentionally an
external dependency because its Arrow output is a runtime community extension;
install it separately or provide it on `PATH`.

### Native CLI (Arrow streaming renderer)

`trd` is a pure Arrow filter: it reads an Arrow IPC stream of per-frame
parameters on stdin and writes an Arrow IPC stream of rendered images on stdout
(trd stream protocol 0.0.1). It never buffers the whole animation — one record
batch is in flight at a time.

Frame parameters are just columnar data, so any tool that emits the input
columns as an Arrow IPC stream can drive the renderer. The example input lives
in [`examples/frames.jsonl`](examples/frames.jsonl) (one JSON object per frame:
`center`, `size`, `theta`). Render it to a GIF with the wrapper script:

```sh
# First enter the project environment:
nix develop

# Then render. On WSL, prefix with WGPU_BACKEND=gl for GPU rendering.
examples/render.sh examples/frames.jsonl out.gif
# examples/render.sh [INPUT.jsonl] [OUTPUT.gif|.webp] [WIDTH] [HEIGHT] [FPS]
```

The Nix shell provides `cargo`, `uv`, and `ffmpeg`; `duckdb` must also be on
`PATH`. The script checks these prerequisites before starting, so a missing tool
cannot cause a misleading downstream Arrow error. Under the hood it is a
fully-piped JSONL -> DuckDB -> trd -> ffmpeg flow (no intermediate files) —
**DuckDB** reads the JSONL, casts the `[x, y]` arrays to fixed-size `FLOAT[2]`
(Arrow `FixedSizeList<f32>[2]`), and streams Arrow IPC to stdout:

```sh
duckdb -c "INSTALL arrow FROM community; LOAD arrow;
  COPY (
    SELECT center::FLOAT[2] AS center, size::FLOAT[2] AS size, theta::FLOAT AS theta
    FROM read_json_auto('examples/frames.jsonl')
  ) TO '/dev/stdout' (FORMAT arrows);" \
  | WGPU_BACKEND=gl cargo run -q -p trd-cli -- --width 256 --height 256 \
  | uv run --with pyarrow --with numpy scripts/encode.py --fps 30 -o out.gif
```

- DuckDB (an external tool) emits the input stream. `FORMAT arrows` (plural) is
  the streaming IPC format. The protocol version metadata is optional, so
  DuckDB's stream is accepted as-is.
- `trd` renders each row to `r,g,b,a` `fixed_shape_tensor<u8>` channels.
- `scripts/encode.py` decodes the tensors and pipes RGBA frames to ffmpeg
  (`.gif` or `.webp` by output extension). On non-WSL GPUs, drop `WGPU_BACKEND=gl`.

### Native window (interactive)

Opens a window and renders the triangle into a live wgpu surface (the desktop
counterpart of the browser wasm target):

```sh
cargo run -p trd-app
```

It honours `WGPU_BACKEND` / `RUST_LOG` like the CLI. Close the window to exit.

### Tests

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace                 # fast; GPU tests are skipped
cargo test --workspace -- --ignored    # GPU-gated render tests
```

### Web (wasm)

For production/CI, build the servable bundle reproducibly with Nix:

```sh
nix build .#web    # Rust core -> wasm-bindgen library -> bun dist/ (in ./result)
nix run   .#web    # serve the built dist/ over HTTP (PORT defaults to 8080)
```

The wasm core is a standard, TypeScript-typed npm package. In the nix build it
is produced by `wasm-bindgen-cli` + `wasm-opt` (`nix build .#trd-wasm`); `web/`
consumes it as the `trd-wasm` dependency and imports it like any other library:

```ts
import init, { start } from "trd-wasm"; // fully typed
```

For local iteration inside `nix develop`, the `web/` scripts use `wasm-pack`
(which emits the same package shape) plus bun:

```sh
cd web
bun run build      # wasm-pack -> pkg, then bun bundles -> web/dist
bun run dev        # dev server; open the printed URL in a WebGPU browser
bun run check      # Biome format-check + lint
bun run format     # Biome auto-format
bun run typecheck  # tsc --noEmit
```

The wasm-pack/wasm-bindgen `web` target is used because bun does not instantiate
the `bundler` target's ESM-imported wasm.
